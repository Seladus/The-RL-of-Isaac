json = require('json')
mod = RegisterMod("isaac-env", 1)
mod.rng = RNG()
mod.host = "127.0.0.1"
mod.port = 6942
mod.connection_initialized = false
mod.first_frame = true
mod.is_resetted = false
mod.is_closed = false
mod.frameskip = 5
mod.frame_fire_delay = 5
mod.use_virtual_controller = true
mod.static_enemies = false

player_state = {
    fire = -1,
    move = -1,
    last_framecount_fire = -1
}

Fire = {
    RIGHT = Vector(10, 0),
    LEFT = Vector(-10, 0),
    TOP = Vector(0, -10),
    DOWN = Vector(0, 10)
}

Move = {
    RIGHT = Vector(10, 0),
    LEFT = Vector(-10, 0),
    TOP = Vector(0, -10),
    DOWN = Vector(0, 10)
}

function resetPlayerState()
    player_state = {
        fire = -1,
        move = -1,
        last_framecount_fire = 0
    }
end

function setPlayerDamages(d)
    Isaac.GetPlayer().Damage = d
end

function updateInGamePlayer()
    local player = Isaac.GetPlayer()
    local game = Game()
    local framecount = game:GetFrameCount()
    -- shoot tear
    if player_state.fire ~= -1 then
        if framecount - player_state.last_framecount_fire > mod.frame_fire_delay then
            if player_state.fire == 0 then
                player:FireTear(player.Position, Fire.DOWN, false, false, true, player, 1)
            elseif player_state.fire == 1 then
                player:FireTear(player.Position, Fire.TOP, false, false, true, player, 1)
            elseif player_state.fire == 2 then
                player:FireTear(player.Position, Fire.LEFT, false, false, true, player, 1)
            elseif player_state.fire == 3 then
                player:FireTear(player.Position, Fire.RIGHT, false, false, true, player, 1)
            end
            player_state.last_framecount_fire = framecount
            print(framecount)
        end
    end
    -- move player
    if player_state.move ~= -1 then
        if player_state.move == 0 then
            player.Velocity = Move.DOWN
        elseif player_state.move == 1 then
            player.Velocity = Move.TOP
        elseif player_state.move == 2 then
            player.Velocity = Move.LEFT
        elseif player_state.move == 3 then
            player.Velocity = Move.RIGHT
        end
    else
        player.Velocity = Vector(0, 0)
    end
end

function setCurse(id)
    Isaac.ExecuteCommand("curse " .. id)
end

function disableHud(game)
    game:GetHUD():SetVisible(false)
end

function setFullHearts()
    Isaac.GetPlayer():SetFullHearts()
end

function setMaxHearts(nb)
    local player = Isaac.GetPlayer()
    player:AddMaxHearts(nb - player:GetMaxHearts(), true)
end

function restart(roomId)
    Isaac.ExecuteCommand("restart")
end

function give_brimstome()
    Isaac.ExecuteCommand("giveitem Brimstone")
end

function removeDoors(room)
    room:RemoveDoor(DoorSlot.DOWN0)
    room:RemoveDoor(DoorSlot.DOWN1)
    room:RemoveDoor(DoorSlot.LEFT0)
    room:RemoveDoor(DoorSlot.LEFT1)
    room:RemoveDoor(DoorSlot.RIGHT0)
    room:RemoveDoor(DoorSlot.RIGHT1)
    room:RemoveDoor(DoorSlot.UP0)
    room:RemoveDoor(DoorSlot.UP1)
end

function warpToRoom(roomId)
    Isaac.ExecuteCommand("goto d." .. roomId)
end

function warpToStage(stage)
    Isaac.ExecuteCommand("stage " .. stage)
end

function spawnMonstro(position)
    -- mod.e = Isaac.Spawn(EntityType.ENTITY_MONSTRO, 1, 0, position, Vector(0, 0), nil)
    mod.e = Game():Spawn(EntityType.ENTITY_MONSTRO, 0, position, Vector(0, 0), nil, 2, 69)
end

function spawnDingle(position)
    -- mod.e = Isaac.Spawn(EntityType.ENTITY_MONSTRO, 1, 0, position, Vector(0, 0), nil)
    mod.e = Game():Spawn(EntityType.ENTITY_DINGLE, 0, position, Vector(0, 0), nil, 0, 69)
end

function spawnGaper(position)
    mod.e = Isaac.Spawn(EntityType.ENTITY_GAPER, 0, 0, position, Vector(0, 0), nil)
end

function generateRandomXYFarFromPlayer(distSq, playerPosition, room)
    local randomPosition
    repeat
        randomPosition = room:GetRandomPosition(0)
    until randomPosition:DistanceSquared(playerPosition) > distSq
    return randomPosition
end

function randomSpawnMobs(nb, distFromPlayer, playerPosition, room, spawnFunc)
    local randomPos
    for i = 1, nb, 1 do
        randomPos = generateRandomXYFarFromPlayer(distFromPlayer ^ 2, playerPosition, room)
        spawnFunc(randomPos)
    end
end

function waitForClientConnection()
    mod.socket = require("socket")
    mod.server = mod.socket.bind(mod.host, mod.port)
    mod.client = mod.server:accept()
    -- local keep = client:setoption("keepalive", true)
    mod.connection_initialized = true
    mod.server:close()
end

function init(game)
    local level = game:GetLevel()
    local room = level:GetCurrentRoom()
    local player = Isaac.GetPlayer()

    resetPlayerState()

    -- disableHud(game)
    -- if in basement 1 warp to basement 2
    if level:GetStageType() == 0 then
        warpToStage(2)
    end
    -- spawn mobs
    -- mod.nb_mob = mod.rng:RandomInt(10) + 1
    mod.nb_mob = 1
    -- spawnMonstro(Vector(room:GetCenterPos().X, room:GetCenterPos().Y + 200))
    randomSpawnMobs(mod.nb_mob, 100, player.Position, room, spawnMonstro)
    randomSpawnMobs(mod.nb_mob, 100, player.Position, room, spawnMonstro)
    -- randomSpawnMobs(mod.nb_mob, 100, player.Position, room, spawnGaper)
    -- close room
    removeDoors(room)
    setMaxHearts(6)
    setFullHearts()
    setPlayerDamages(20) -- setPlayerDamages(10)

    mod.current_gamestate = {
        isDead = false,
        lifeCount = 0,
        damagesTaken = 0,
        mobDamagesTaken = 0,
        mobKilled = 0,
        done = false
    }
    mod:updateGamestate()
    mod.previous_gamestate = mod.current_gamestate
end

function mod:handleResetMessage(parsed_msg)
    mod.first_frame = true
    mod.is_resetted = true
    mod.use_virtual_controller = parsed_msg.use_virtual_controller
    print(mod.use_virtual_controller)
    print("restart")
    restart()
    return false
end

function mod:handleStepMessage(parsed_msg)
    if parsed_msg.player_move ~= nil and parsed_msg.player_fire ~= nil then
        player_state.fire = parsed_msg.player_fire
        player_state.move = parsed_msg.player_move
    end
    return true
end

function mod:handleCloseMessage(parsed_msg)
    mod.is_closed = true
    return false
end

function mod:updateGamestate()
    mod.current_gamestate.enemy_positions = {}
    mod.current_gamestate.enemy_healths = {}
    mod.current_gamestate.enemy_velocities = {}
    mod.current_gamestate.tears_positions = {}
    mod.current_gamestate.tears_velocities = {}
    mod.current_gamestate.enemy_projectiles_positions = {}
    mod.current_gamestate.enemy_projectiles_velocities = {}
    -- separate entities
    for i, entity in ipairs(Isaac.GetRoomEntities()) do
        -- if is enemy
        if entity:IsEnemy() then
            table.insert(mod.current_gamestate.enemy_positions, {entity.Position.X, entity.Position.Y})
            table.insert(mod.current_gamestate.enemy_velocities, {entity.Velocity.X, entity.Velocity.Y})
            table.insert(mod.current_gamestate.enemy_healths, entity.HitPoints)
        end
        -- if is player
        if entity:ToPlayer() then
            mod.current_gamestate.player_position = {entity.Position.X, entity.Position.Y}
            mod.current_gamestate.player_velocity = {entity.Velocity.X, entity.Velocity.Y}
        end
        -- if is tear of player
        if entity.Type == EntityType.ENTITY_TEAR then
            table.insert(mod.current_gamestate.tears_positions, {entity.Position.X, entity.Position.Y})
            table.insert(mod.current_gamestate.tears_velocities, {entity.Velocity.X, entity.Velocity.Y})
        end
        -- if is tear of ennemy
        if entity.Type == EntityType.ENTITY_PROJECTILE then
            table.insert(mod.current_gamestate.enemy_projectiles_positions, {entity.Position.X, entity.Position.Y})
            table.insert(mod.current_gamestate.enemy_projectiles_velocities, {entity.Velocity.X, entity.Velocity.Y})
        end
    end
    -- get if room is done
    mod.current_gamestate.done = mod.current_gamestate.done or Isaac.CountEnemies() == 0
end

function mod:handleMessage(msg, err)
    if msg == nil then
        -- mod.connection_initialized = false
        return false
    end
    local parsed_msg = json.decode(msg)
    if parsed_msg.action == "reset" then
        return mod:handleResetMessage(parsed_msg)
    elseif parsed_msg.action == "step" then
        return mod:handleStepMessage(parsed_msg)
    elseif parsed_msg.action == "close" then
        return mod:handleCloseMessage(parsed_msg)
    end
end

function mod:onUpdate()
    -- Isaac.GetPlayer():SetColor(Color(1, 1, 1, 1, 255, 255, 255), 15, 1, false, false)
    local game = Game()
    local framecount = game:GetFrameCount()

    if not mod.is_closed then
        if not mod.connection_initialized then
            waitForClientConnection()
        end

        if mod.first_frame then
            print("first frame")
            -- init environment
            init(game)
            if mod.is_resetted then
                mod.client:send(json.encode(mod.current_gamestate))
                mod.is_resetted = false
            end
            mod.first_frame = false
        else
            if not mod.use_virtual_controller then
                updateInGamePlayer()
            end
            if ((framecount + 1) % mod.frameskip == 0) then
                -- Wait for client instructions : Reset
                local line, error = mod.client:receive()
                -- Handle message received
                if mod:handleMessage(line, error) then
                    mod:updateGamestate()
                    -- Return current game state : Done or not + other infos if needed
                    mod.client:send(json.encode(mod.current_gamestate))
                    mod.previous_gamestate = mod.current_gamestate
                end
            end
        end
    end
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.onUpdate)

function mod:onEntityKill(entity)
    -- if player dies
    if entity:ToPlayer() then
        mod.current_gamestate.isDead = true
        mod.current_gamestate.done = true
    else
        mod.current_gamestate.mobKilled = mod.current_gamestate.mobKilled + 1
    end
end

mod:AddCallback(ModCallbacks.MC_POST_ENTITY_KILL, mod.onEntityKill)

function mod:onDamageTaken(entity, amount, flags, dealer, countdown)
    -- if player takes damage
    if entity:ToPlayer() then
        mod.current_gamestate.damagesTaken = mod.current_gamestate.damagesTaken + 1
    else
        mod.current_gamestate.mobDamagesTaken = mod.current_gamestate.mobDamagesTaken + 1
    end
end

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, mod.onDamageTaken)

function mod:onNPCUpdate(entity)
    if entity:ToPlayer() == nil and mod.static_enemies then
        entity.Velocity = Vector(0, 0)
    end
end

mod:AddCallback(ModCallbacks.MC_NPC_UPDATE, mod.onNPCUpdate)

