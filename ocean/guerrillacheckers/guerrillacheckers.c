#include "guerrillacheckers.h"

#define GC_DEMO_NOOP -1

static void gc_demo_allocate(GuerrillaCheckers* env) {
    env->observations = (uint8_t*)calloc(GC_OBS_SIZE, sizeof(uint8_t));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    env->action_mask = (unsigned char*)calloc(GC_ACTIONS, sizeof(unsigned char));
}

static void gc_demo_free(GuerrillaCheckers* env) {
    c_close(env);
    free(env->action_mask);
    free(env->terminals);
    free(env->rewards);
    free(env->actions);
    free(env->observations);
}

static int gc_demo_mouse_coin_cell(GuerrillaCheckers* env, Vector2 mouse) {
    if (env->client == NULL) return -1;
    int cell = env->client->cell;
    int x = (int)mouse.x / cell;
    int y = (int)mouse.y / cell;
    if (!gc_valid_coin_xy(x, y)) return -1;
    return gc_coin_pos(x, y);
}

static int gc_demo_mouse_guerrilla_cell(GuerrillaCheckers* env, Vector2 mouse) {
    if (env->client == NULL) return -1;
    int cell = env->client->cell;
    int best = -1;
    float best_dist2 = (float)(cell * cell);
    float limit = (float)(cell * cell) * 0.18f;
    for (int y = 0; y < GC_G_H; y++) {
        for (int x = 0; x < GC_G_W; x++) {
            float dx = mouse.x - (float)((x + 1) * cell);
            float dy = mouse.y - (float)((y + 1) * cell);
            float dist2 = dx * dx + dy * dy;
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best = gc_g_pos(x, y);
            }
        }
    }
    return best_dist2 <= limit ? best : -1;
}

static int gc_demo_first_has_legal_action(GuerrillaCheckers* env, int first) {
    if (first < 0) return 0;
    for (int dir = 0; dir < 4; dir++) {
        int action = first * 4 + dir;
        if (gc_action_allowed_by_current_mask(env, action)) return 1;
    }
    return 0;
}

static int gc_demo_guerrilla_action(int first, int second) {
    static const int dirs[4] = {2, 3, 0, 1};
    for (int dir = 0; dir < 4; dir++) {
        if (gc_g_neighbor(first, dirs[dir]) == second) return first * 4 + dir;
    }
    return GC_DEMO_NOOP;
}

static int gc_demo_coin_action(int src, int dst) {
    for (int dir = 0; dir < 4; dir++) {
        if (gc_coin_neighbor(src, dir) == dst) return src * 4 + dir;
    }
    return GC_DEMO_NOOP;
}

static int gc_demo_human_action(GuerrillaCheckers* env, int* selected) {
    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) return GC_DEMO_NOOP;

    Vector2 mouse = GetMousePosition();
    if (env->player_to_move == GC_GUERRILLA) {
        int pos = gc_demo_mouse_guerrilla_cell(env, mouse);
        if (pos < 0) return GC_DEMO_NOOP;

        if (*selected < 0 || !gc_demo_first_has_legal_action(env, *selected)) {
            if (gc_demo_first_has_legal_action(env, pos)) *selected = pos;
            return GC_DEMO_NOOP;
        }

        int action = gc_demo_guerrilla_action(*selected, pos);
        if (gc_action_allowed_by_current_mask(env, action)) return action;
        if (gc_demo_first_has_legal_action(env, pos)) *selected = pos;
        return GC_DEMO_NOOP;
    }

    int pos = gc_demo_mouse_coin_cell(env, mouse);
    if (pos < 0) return GC_DEMO_NOOP;

    if (*selected < 0 || !env->coin_cells[*selected]) {
        if (env->coin_cells[pos]) *selected = pos;
        return GC_DEMO_NOOP;
    }

    int action = gc_demo_coin_action(*selected, pos);
    if (gc_action_allowed_by_current_mask(env, action)) return action;
    if (env->coin_cells[pos]) *selected = pos;
    return GC_DEMO_NOOP;
}

static int gc_demo_auto_action(GuerrillaCheckers* env) {
    if (env->game_over) return 0;  // c_step will reset terminal demos.
    return gc_bot_action(env);
}

static void demo(void) {
    GuerrillaCheckers env = {0};
    env.num_agents = 1;
    env.max_episode_length = 256;
    env.render_fps = 12;
    env.selfplay = 1;
    env.side_cfg = 0;
    env.opponent = GC_BOT_GREEDY;
    env.mcts_iterations = 256;
    env.mcts_exploration = GC_MCTS_DEFAULT_EXPLORATION;
    env.mcts_rollout = GC_MCTS_ROLLOUT_GREEDY;
    env.rng = 1234u;

    gc_demo_allocate(&env);
    c_reset(&env);
    c_render(&env);

    int tick = 0;
    int selected = GC_DEMO_NOOP;
    int selected_player = GC_NONE;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            selected = GC_DEMO_NOOP;
            selected_player = GC_NONE;
        }

        if (selected_player != env.player_to_move) {
            selected = GC_DEMO_NOOP;
            selected_player = env.player_to_move;
        }

        int action = GC_DEMO_NOOP;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            action = gc_demo_human_action(&env, &selected);
        } else if (tick % 12 == 0) {
            action = gc_demo_auto_action(&env);
        }

        tick = (tick + 1) % 12;
        if (action != GC_DEMO_NOOP) {
            env.actions[0] = (float)action;
            c_step(&env);
            selected = GC_DEMO_NOOP;
            selected_player = env.player_to_move;
        }

        c_render(&env);
    }

    gc_demo_free(&env);
}

int main(void) {
    demo();
    return 0;
}
