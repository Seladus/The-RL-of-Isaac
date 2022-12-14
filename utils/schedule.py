from typing import Tuple, List


class Schedule:
    def __call__(self, x):
        raise NotImplementedError()


class Flat(Schedule):
    def __init__(self, value):
        self.__value = value

    def __call__(self, x):
        return self.__value

    def __str__(self):
        return f"Schedule({self.__value})"


class Dynamic(Schedule):
    def __init__(self, value):
        self.__value = value

    def __call__(self, x):
        return self.__value

    def update(self, value):
        self.__value = value

    def __str__(self):
        return "Dynamic"


class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class Piecewise(Schedule):
    """
    ## Piecewise schedule
    """

    def __init__(
        self, endpoints: List[Tuple[float, float]], outside_value: float = None
    ):
        """
        ### Initialize
        `endpoints` is list of pairs `(x, y)`.
         The values between endpoints are linearly interpolated.
        `y` values outside the range covered by `x` are
        `outside_value`.
        """

        # `(x, y)` pairs should be sorted
        indexes = [e[0] for e in endpoints]
        assert indexes == sorted(indexes)

        self._outside_value = outside_value
        self._endpoints = endpoints

    def __call__(self, x):
        """
        ### Find `y` for given `x`
        """

        # iterate through each segment
        for (x1, y1), (x2, y2) in zip(self._endpoints[:-1], self._endpoints[1:]):
            # interpolate if `x` is within the segment
            if x1 <= x < x2:
                dx = float(x - x1) / (x2 - x1)
                return y1 + dx * (y2 - y1)

        # return outside value otherwise
        return self._outside_value

    def __str__(self):
        endpoints = ", ".join([f"({e[0]}, {e[1]})" for e in self._endpoints])
        return f"Schedule[{endpoints}, {self._outside_value}]"


class RelativePiecewise(Piecewise):
    def __init__(self, relative_endpoits: List[Tuple[float, float]], total_steps: int):
        endpoints = []
        for e in relative_endpoits:
            index = int(total_steps * e[0])
            assert index >= 0
            endpoints.append((index, e[1]))

        super().__init__(endpoints, outside_value=relative_endpoits[-1][1])
