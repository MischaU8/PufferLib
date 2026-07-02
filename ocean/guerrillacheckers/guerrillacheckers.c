// Standalone Guerrilla Checkers client: a menu picks Human or AI (with a
// per-side level) for each side, then play is fully mouse-driven (click a
// piece or point, then a destination). AI vs AI can also run as a fast
// tournament with a running win tally. Works the same in native and
// Emscripten web builds.
#include "guerrillacheckers.h"
#include "puffernet.h"

#define GC_DEMO_NOOP -1

enum {
    GC_UI_MENU = 0,
    GC_UI_PLAY = 1,
    GC_UI_TOURNEY = 2,
};

enum {
    GC_CTRL_HUMAN = 0,
    GC_CTRL_AI = 1,
};

typedef struct {
    const char* label;
    int opponent;
    int mcts_iterations;
} GcDemoLevel;

#define GC_DEMO_NET_BOT -1  // sentinel opponent id: puffernet policy

static const GcDemoLevel gc_demo_levels[] = {
    {"1", GC_BOT_RANDOM, 0},
    {"2", GC_BOT_GREEDY, 0},
    {"3", GC_DEMO_NET_BOT, 0},
    {"4", GC_BOT_MCTS, 2000},
    {"5", GC_BOT_MCTS, 10000},
};
#define GC_DEMO_LEVEL_COUNT 5
#define GC_DEMO_LEVEL_NET 2      // index of the PUFFER NN entry
#define GC_DEMO_LEVEL_MCTS_2K 3  // default when the net weights are unavailable
static const char* gc_demo_level_legend =
    "1 RANDOM   2 GREEDY   3 PUFFER NN   4 MCTS 2K   5 MCTS 10K";
static const char* gc_demo_level_legend_no_net =
    "1 RANDOM   2 GREEDY   4 MCTS 2K   5 MCTS 10K";

// Trained policy (see export_weights.py): the default PufferLib policy from
// config/guerrillacheckers.ini, run with puffernet. One instance per side so
// the recurrent MinGRU state never mixes the two players' turns.
#define GC_DEMO_NET_HIDDEN 128
#define GC_DEMO_NET_LAYERS 2
#define GC_DEMO_NET_WEIGHTS "resources/guerrillacheckers/guerrillacheckers_weights.bin"

typedef struct {
    Weights* weights;
    Affine* encoder;
    Affine* decoder;
    MinGRU* gru;
} GcDemoNet;

static GcDemoNet gc_demo_nets[3];  // indexed by GC_GUERRILLA / GC_COIN
static int gc_demo_net_loaded = 0;

#define GC_DEMO_AI_WAIT 20        // frames between an action and the AI's reply
#define GC_DEMO_OVER_WAIT 45      // frames before a click can leave the game-over screen
#define GC_DEMO_TOURNEY_BUDGET 0.012  // seconds of simulation per rendered frame
#define GC_DEMO_TRAIL_MAX 8

// Ghost trail of the last completed turn, so the opponent's move stays
// readable after it lands. A coin capture chain accumulates into one trail,
// with every square the coin visited kept in coin_path.
typedef struct {
    int side;  // GC_NONE while empty
    int placed[2];
    int placed_count;
    int coin_path[GC_DEMO_TRAIL_MAX + 2];
    int coin_path_count;
    int captured_g[GC_DEMO_TRAIL_MAX];
    int captured_g_count;
    int captured_coins[GC_DEMO_TRAIL_MAX];
    int captured_coins_count;
} GcDemoTrail;

typedef struct {
    int games;
    int guerrilla_wins;
    int coin_wins;
} GcDemoTally;

typedef struct {
    int mode;
    int ctrl[3];      // indexed by GC_GUERRILLA / GC_COIN
    int ai_level[3];  // ditto
    int selected;  // guerrilla point or coin square awaiting a destination click
    int ai_wait;
    int over_wait;
    int paused;  // tournament simulation paused (spacebar)
    GcDemoTrail trail;
    GcDemoTally tally;
} GcDemoUi;

static const Color GC_DEMO_BG = {18, 24, 28, 255};
static const Color GC_DEMO_TEXT = {190, 204, 208, 255};
static const Color GC_DEMO_DIM = {104, 126, 132, 255};
static const Color GC_DEMO_GUERRILLA = {206, 72, 72, 255};
static const Color GC_DEMO_COIN = {232, 198, 83, 255};
static const Color GC_DEMO_SELECT = {255, 255, 255, 230};
static const Color GC_DEMO_G_HINT = {206, 72, 72, 160};      // ghost guerrilla stones
static const Color GC_DEMO_G_HINT_DIM = {206, 72, 72, 110};  // legal first placements
static const Color GC_DEMO_C_HINT = {255, 240, 170, 230};    // coin move markers
static const Color GC_DEMO_C_SQUARE = {232, 198, 83, 60};    // coin destination squares
static const Color GC_DEMO_TRAIL = {255, 255, 255, 200};     // "just moved here" dots
static const Color GC_DEMO_G_GHOST = {206, 72, 72, 190};     // captured guerrilla X
static const Color GC_DEMO_C_GHOST = {232, 198, 83, 150};    // coin path lines
static const Color GC_DEMO_C_CAPTURED = {232, 198, 83, 190}; // captured coin X
static const Color GC_DEMO_C_ORIGIN = {232, 198, 83, 90};    // coin origin ghost piece

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

// Both sides encode actions as pos * 4 + dir, so one check covers guerrilla
// first-placement points and coin source squares.
static int gc_demo_pos_has_legal_action(GuerrillaCheckers* env, int pos) {
    if (pos < 0) return 0;
    for (int dir = 0; dir < 4; dir++) {
        if (gc_action_allowed_by_current_mask(env, pos * 4 + dir)) return 1;
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
        if (pos == *selected) {
            *selected = GC_DEMO_NOOP;
            return GC_DEMO_NOOP;
        }
        if (*selected >= 0) {
            int action = gc_demo_guerrilla_action(*selected, pos);
            if (gc_action_allowed_by_current_mask(env, action)) return action;
        }
        if (gc_demo_pos_has_legal_action(env, pos)) *selected = pos;
        return GC_DEMO_NOOP;
    }

    int pos = gc_demo_mouse_coin_cell(env, mouse);
    if (pos < 0) return GC_DEMO_NOOP;
    if (pos == *selected) {
        // A capture chain must be continued with the same piece: keep it selected.
        if (!env->coin_must_capture) *selected = GC_DEMO_NOOP;
        return GC_DEMO_NOOP;
    }
    if (*selected >= 0) {
        int action = gc_demo_coin_action(*selected, pos);
        if (gc_action_allowed_by_current_mask(env, action)) return action;
    }
    if (gc_demo_pos_has_legal_action(env, pos)) *selected = pos;
    return GC_DEMO_NOOP;
}

static void gc_demo_trail_update(GcDemoUi* ui, GuerrillaCheckers* env, int actor,
        const uint8_t* coins_before, const uint8_t* g_before) {
    GcDemoTrail* trail = &ui->trail;

    int src = -1;
    int dst = -1;
    for (int i = 0; i < GC_COIN_CELLS; i++) {
        if (coins_before[i] && !env->coin_cells[i]) src = i;
        if (!coins_before[i] && env->coin_cells[i]) dst = i;
    }

    // Consecutive coin actions from the same square chain into one trail
    // (forced multi-captures); anything else starts a fresh trail.
    int chain = actor == GC_COIN && trail->side == GC_COIN &&
        src >= 0 && trail->coin_path_count > 0 &&
        trail->coin_path[trail->coin_path_count - 1] == src;
    if (!chain) {
        memset(trail, 0, sizeof(*trail));
        trail->side = actor;
    }

    if (actor == GC_COIN) {
        if (!chain && src >= 0) trail->coin_path[trail->coin_path_count++] = src;
        if (dst >= 0 && trail->coin_path_count < GC_DEMO_TRAIL_MAX + 2) {
            trail->coin_path[trail->coin_path_count++] = dst;
        }
        for (int i = 0; i < GC_G_CELLS; i++) {
            if (g_before[i] && !env->guerrilla_cells[i] &&
                    trail->captured_g_count < GC_DEMO_TRAIL_MAX) {
                trail->captured_g[trail->captured_g_count++] = i;
            }
        }
        return;
    }

    for (int i = 0; i < GC_G_CELLS; i++) {
        if (!g_before[i] && env->guerrilla_cells[i] && trail->placed_count < 2) {
            trail->placed[trail->placed_count++] = i;
        }
    }
    for (int i = 0; i < GC_COIN_CELLS; i++) {
        if (coins_before[i] && !env->coin_cells[i] &&
                trail->captured_coins_count < GC_DEMO_TRAIL_MAX) {
            trail->captured_coins[trail->captured_coins_count++] = i;
        }
    }
}

// Apply one legal action, advance the turn, and record the ghost trail.
static void gc_demo_apply(GuerrillaCheckers* env, GcDemoUi* ui, int action) {
    uint8_t coins_before[GC_COIN_CELLS];
    uint8_t g_before[GC_G_CELLS];
    memcpy(coins_before, env->coin_cells, sizeof(coins_before));
    memcpy(g_before, env->guerrilla_cells, sizeof(g_before));
    int actor = env->player_to_move;

    gc_apply_action(env, action);
    env->tick++;
    gc_prepare_turn(env);
    gc_demo_trail_update(ui, env, actor, coins_before, g_before);
}

static int gc_demo_net_init(GcDemoNet* net, const char* path) {
    net->weights = load_weights(path);
    if (net->weights == NULL) return 0;
    net->encoder = make_affine(net->weights, 1, GC_OBS_SIZE, GC_DEMO_NET_HIDDEN);
    net->decoder = make_affine(net->weights, 1, GC_DEMO_NET_HIDDEN, GC_ACTIONS);
    net->gru = make_mingru(net->weights, 1, GC_DEMO_NET_HIDDEN, GC_DEMO_NET_LAYERS);
    return 1;
}

// Clear both sides' recurrent state at the start of every game.
static void gc_demo_net_reset(void) {
    if (!gc_demo_net_loaded) return;
    for (int side = GC_GUERRILLA; side <= GC_COIN; side++) {
        MinGRU* gru = gc_demo_nets[side].gru;
        memset(gru->state, 0,
            (size_t)gru->num_layers * gru->hidden_size * sizeof(float));
    }
}

static int gc_demo_net_action(GuerrillaCheckers* env) {
    GcDemoNet* net = &gc_demo_nets[env->player_to_move];
    gc_compute_observations(env);
    float obs[GC_OBS_SIZE];
    for (int i = 0; i < GC_OBS_SIZE; i++) {
        obs[i] = (float)env->observations[i];
    }
    affine(net->encoder, obs);
    mingru(net->gru, net->encoder->output);
    affine(net->decoder, net->gru->output);
    float* logits = net->decoder->output;

    // Sample from the softmax over legal actions, matching the masked
    // sampling the policy was trained with.
    float max_logit = -1e30f;
    for (int a = 0; a < GC_ACTIONS; a++) {
        if (env->action_mask[a] && logits[a] > max_logit) max_logit = logits[a];
    }
    float probs[GC_ACTIONS];
    float total = 0.0f;
    for (int a = 0; a < GC_ACTIONS; a++) {
        probs[a] = env->action_mask[a] ? expf(logits[a] - max_logit) : 0.0f;
        total += probs[a];
    }
    float r = (float)gc_rand(env) / ((float)RAND_MAX + 1.0f) * total;
    int last_legal = GC_DEMO_NOOP;
    for (int a = 0; a < GC_ACTIONS; a++) {
        if (probs[a] <= 0.0f) continue;
        last_legal = a;
        r -= probs[a];
        if (r <= 0.0f) return a;
    }
    return last_legal;
}

static int gc_demo_button(Rectangle rect, const char* label, int active) {
    Vector2 mouse = GetMousePosition();
    int hover = CheckCollisionPointRec(mouse, rect);
    Color fill = active ? (Color){96, 44, 44, 255} :
        hover ? (Color){54, 70, 78, 255} : (Color){36, 48, 54, 255};
    DrawRectangleRec(rect, fill);
    DrawRectangleLinesEx(rect, 2.0f, active ? GC_DEMO_GUERRILLA : GC_DEMO_DIM);
    int size = 18;
    int width = MeasureText(label, size);
    DrawText(label, (int)(rect.x + (rect.width - (float)width) / 2.0f),
        (int)(rect.y + (rect.height - (float)size) / 2.0f), size,
        active ? RAYWHITE : GC_DEMO_TEXT);
    return hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
}

static void gc_demo_start_game(GuerrillaCheckers* env, GcDemoUi* ui) {
    // Vary the seed per game so bot play differs between runs.
    env->rng ^= (unsigned int)(GetTime() * 1000.0) | 1u;
    c_reset(env);
    ui->selected = GC_DEMO_NOOP;
    ui->ai_wait = GC_DEMO_AI_WAIT;
    ui->over_wait = 0;
    ui->paused = 0;
    memset(&ui->trail, 0, sizeof(ui->trail));
    gc_demo_net_reset();
}

static void gc_demo_render_menu(GuerrillaCheckers* env, GcDemoUi* ui) {
    Client* client = env->client;
    BeginDrawing();
    ClearBackground(GC_DEMO_BG);

    const char* title = "GUERRILLA CHECKERS";
    DrawText(title, (client->width - MeasureText(title, 32)) / 2, 56, 32, RAYWHITE);

    static const struct { const char* name; int side; float y; } rows[] = {
        {"GUERRILLA", GC_GUERRILLA, 128.0f},
        {"COIN", GC_COIN, 232.0f},
    };
    for (int r = 0; r < 2; r++) {
        int side = rows[r].side;
        float y = rows[r].y;
        DrawText(rows[r].name, 64, (int)y + 8, 20,
            side == GC_GUERRILLA ? GC_DEMO_GUERRILLA : GC_DEMO_COIN);
        if (gc_demo_button((Rectangle){256, y, 112, 36}, "HUMAN",
                ui->ctrl[side] == GC_CTRL_HUMAN)) {
            ui->ctrl[side] = GC_CTRL_HUMAN;
        }
        if (gc_demo_button((Rectangle){384, y, 112, 36}, "AI",
                ui->ctrl[side] == GC_CTRL_AI)) {
            ui->ctrl[side] = GC_CTRL_AI;
        }
        if (ui->ctrl[side] == GC_CTRL_AI) {
            DrawText("LEVEL", 186, (int)y + 52, 16, GC_DEMO_DIM);
            int shown = 0;
            for (int i = 0; i < GC_DEMO_LEVEL_COUNT; i++) {
                if (i == GC_DEMO_LEVEL_NET && !gc_demo_net_loaded) continue;
                Rectangle rect = {256 + (float)shown * 52.0f, y + 44.0f, 44, 32};
                shown++;
                if (gc_demo_button(rect, gc_demo_levels[i].label,
                        ui->ai_level[side] == i)) {
                    ui->ai_level[side] = i;
                }
            }
        }
    }

    const char* legend = gc_demo_net_loaded ? gc_demo_level_legend :
        gc_demo_level_legend_no_net;
    DrawText(legend, (client->width - MeasureText(legend, 14)) / 2, 340, 14,
        GC_DEMO_DIM);

    int both_ai = ui->ctrl[GC_GUERRILLA] == GC_CTRL_AI &&
        ui->ctrl[GC_COIN] == GC_CTRL_AI;
    if (both_ai) {
        if (gc_demo_button((Rectangle){110, 400, 150, 48}, "PLAY", 0)) {
            gc_demo_start_game(env, ui);
            ui->mode = GC_UI_PLAY;
        }
        if (gc_demo_button((Rectangle){284, 400, 182, 48}, "TOURNAMENT", 0)) {
            memset(&ui->tally, 0, sizeof(ui->tally));
            gc_demo_start_game(env, ui);
            ui->mode = GC_UI_TOURNEY;
        }
    } else if (gc_demo_button((Rectangle){208, 400, 160, 48}, "PLAY", 0)) {
        gc_demo_start_game(env, ui);
        ui->mode = GC_UI_PLAY;
    }

    const char* help = "CLICK A PIECE OR POINT, THEN A DESTINATION";
    DrawText(help, (client->width - MeasureText(help, 16)) / 2,
        client->height - 56, 16, GC_DEMO_DIM);
    const char* credits = "Game design by Brian Train   -   Code by MischaU8";
    DrawText(credits, (client->width - MeasureText(credits, 16)) / 2,
        client->height - 28, 16, GC_DEMO_DIM);
    EndDrawing();
}

static void gc_demo_draw_x(Vector2 center, float size, Color color) {
    DrawLineEx((Vector2){center.x - size, center.y - size},
        (Vector2){center.x + size, center.y + size}, 2.5f, color);
    DrawLineEx((Vector2){center.x - size, center.y + size},
        (Vector2){center.x + size, center.y - size}, 2.5f, color);
}

static Vector2 gc_demo_coin_center(int pos, int cell) {
    return (Vector2){(float)(gc_coin_x(pos) * cell + cell / 2),
        (float)(gc_coin_y(pos) * cell + cell / 2)};
}

static void gc_demo_render_trail(GuerrillaCheckers* env, GcDemoUi* ui) {
    GcDemoTrail* trail = &ui->trail;
    if (trail->side == GC_NONE) return;
    int cell = env->client->cell;

    // Newly placed guerrilla stones get a small "just moved" dot, not a ring
    // (rings mean "selectable" in the move hints).
    for (int i = 0; i < trail->placed_count; i++) {
        Vector2 center = {(float)((gc_g_x(trail->placed[i]) + 1) * cell),
            (float)((gc_g_y(trail->placed[i]) + 1) * cell)};
        DrawCircleV(center, cell * 0.07f, GC_DEMO_TRAIL);
    }
    for (int i = 0; i < trail->captured_g_count; i++) {
        Vector2 center = {(float)((gc_g_x(trail->captured_g[i]) + 1) * cell),
            (float)((gc_g_y(trail->captured_g[i]) + 1) * cell)};
        gc_demo_draw_x(center, cell * 0.10f, GC_DEMO_G_GHOST);
    }
    for (int i = 0; i < trail->captured_coins_count; i++) {
        gc_demo_draw_x(gc_demo_coin_center(trail->captured_coins[i], cell),
            cell * 0.14f, GC_DEMO_C_CAPTURED);
    }

    if (trail->coin_path_count > 0) {
        // Ghost piece at the origin, lines through every square the coin
        // visited, and a "just moved" dot on its final position.
        DrawCircleV(gc_demo_coin_center(trail->coin_path[0], cell), cell * 0.28f,
            GC_DEMO_C_ORIGIN);
        for (int i = 0; i + 1 < trail->coin_path_count; i++) {
            DrawLineEx(gc_demo_coin_center(trail->coin_path[i], cell),
                gc_demo_coin_center(trail->coin_path[i + 1], cell), 3.0f,
                GC_DEMO_C_GHOST);
        }
        if (trail->coin_path_count > 1) {
            DrawCircleV(gc_demo_coin_center(
                trail->coin_path[trail->coin_path_count - 1], cell),
                cell * 0.07f, GC_DEMO_TRAIL);
        }
    }
}

static void gc_demo_render_hints(GuerrillaCheckers* env, GcDemoUi* ui) {
    int cell = env->client->cell;
    if (env->game_over || ui->ctrl[env->player_to_move] != GC_CTRL_HUMAN) return;

    if (env->player_to_move == GC_GUERRILLA) {
        if (ui->selected >= 0) {
            // First placement pending: draw it as a ghost stone with a
            // selection ring, and ghost dots on the legal second points.
            Vector2 center = {(float)((gc_g_x(ui->selected) + 1) * cell),
                (float)((gc_g_y(ui->selected) + 1) * cell)};
            DrawCircleV(center, cell * 0.20f, GC_DEMO_G_HINT);
            DrawRing(center, cell * 0.20f, cell * 0.20f + 3.0f, 0, 360, 32,
                GC_DEMO_SELECT);
            for (int dir = 0; dir < 4; dir++) {
                int action = ui->selected * 4 + dir;
                if (!gc_action_allowed_by_current_mask(env, action)) continue;
                int first;
                int second;
                gc_decode_guerrilla_action(action, &first, &second);
                if (second < 0) continue;
                DrawCircle((gc_g_x(second) + 1) * cell, (gc_g_y(second) + 1) * cell,
                    cell * 0.13f, GC_DEMO_G_HINT);
            }
        } else if (env->guerrilla_count > 0) {
            // Skip the hints on the opening move: every point is legal and
            // 49 dots just light the whole board up.
            for (int pos = 0; pos < GC_G_CELLS; pos++) {
                if (!gc_demo_pos_has_legal_action(env, pos)) continue;
                DrawCircle((gc_g_x(pos) + 1) * cell, (gc_g_y(pos) + 1) * cell,
                    cell * 0.09f, GC_DEMO_G_HINT_DIM);
            }
        }
        return;
    }

    for (int src = 0; src < GC_COIN_CELLS; src++) {
        if (src == ui->selected || !env->coin_cells[src]) continue;
        if (!gc_demo_pos_has_legal_action(env, src)) continue;
        Vector2 center = {(float)(gc_coin_x(src) * cell + cell / 2),
            (float)(gc_coin_y(src) * cell + cell / 2)};
        DrawRing(center, cell * 0.30f, cell * 0.30f + 3.0f, 0, 360, 32, GC_DEMO_C_HINT);
    }
    if (ui->selected >= 0) {
        Vector2 center = {(float)(gc_coin_x(ui->selected) * cell + cell / 2),
            (float)(gc_coin_y(ui->selected) * cell + cell / 2)};
        DrawRing(center, cell * 0.30f, cell * 0.30f + 4.0f, 0, 360, 32, GC_DEMO_SELECT);
        for (int dir = 0; dir < 4; dir++) {
            if (!gc_action_allowed_by_current_mask(env, ui->selected * 4 + dir)) continue;
            int dst = gc_coin_neighbor(ui->selected, dir);
            int dx = gc_coin_x(dst) * cell;
            int dy = gc_coin_y(dst) * cell;
            DrawRectangle(dx + 3, dy + 3, cell - 6, cell - 6, GC_DEMO_C_SQUARE);
            DrawCircle(dx + cell / 2, dy + cell / 2, cell * 0.13f, GC_DEMO_C_HINT);
        }
    }
}

static void gc_demo_render_paused(Client* client) {
    const char* paused = "PAUSED";
    int width = MeasureText(paused, 16);
    int x = (client->width - width) / 2;
    DrawRectangle(x - 10, 8, width + 20, 28, (Color){0, 0, 0, 170});
    DrawText(paused, x, 14, 16, RAYWHITE);
}

static void gc_demo_render_supply(GuerrillaCheckers* env, int bar_y) {
    const char* counts = TextFormat("GUERRILLAS %d",
        GC_MAX_GUERRILLAS - env->guerrilla_count);
    DrawText(counts, env->client->width - 96 - MeasureText(counts, 16), bar_y + 16,
        16, GC_DEMO_GUERRILLA);
}

// Returns 1 when the MENU button was clicked.
static int gc_demo_render_play(GuerrillaCheckers* env, GcDemoUi* ui) {
    Client* client = env->client;
    int cell = client->cell;
    BeginDrawing();
    ClearBackground(GC_DEMO_BG);
    gc_render_board(env);
    gc_demo_render_trail(env, ui);
    gc_demo_render_hints(env, ui);

    int bar_y = GC_BOARD_H * cell;
    if (env->game_over) {
        Color winner_color = env->winner == GC_GUERRILLA ? GC_DEMO_GUERRILLA : GC_DEMO_COIN;
        const char* winner = env->winner == GC_GUERRILLA ? "GUERRILLA WINS" : "COIN WINS";
        DrawRectangle(0, cell * 3 - 12, client->width, cell + 60, (Color){0, 0, 0, 170});
        DrawText(winner, (client->width - MeasureText(winner, 32)) / 2,
            cell * 3 + 8, 32, winner_color);
        const char* again = "CLICK ANYWHERE FOR MENU";
        DrawText(again, (client->width - MeasureText(again, 16)) / 2,
            cell * 3 + 52, 16, GC_DEMO_TEXT);
        DrawText(winner, 12, bar_y + 13, 22, winner_color);
    } else {
        int side = env->player_to_move;
        const char* name = side == GC_GUERRILLA ? "GUERRILLA" : "COIN";
        const char* verb = ui->ctrl[side] != GC_CTRL_AI ? "TO MOVE" :
            ui->paused ? "PAUSED" : "THINKING...";
        DrawText(TextFormat("%s %s", name, verb), 12, bar_y + 13, 22,
            side == GC_GUERRILLA ? GC_DEMO_GUERRILLA : GC_DEMO_COIN);
    }

    if (ui->paused) gc_demo_render_paused(client);
    gc_demo_render_supply(env, bar_y);
    int menu_clicked = gc_demo_button(
        (Rectangle){(float)client->width - 84.0f, (float)bar_y + 8.0f, 72, 32}, "MENU", 0);
    EndDrawing();
    return menu_clicked;
}

// Returns 1 when the MENU button was clicked.
static int gc_demo_render_tourney(GuerrillaCheckers* env, GcDemoUi* ui) {
    Client* client = env->client;
    int cell = client->cell;
    BeginDrawing();
    ClearBackground(GC_DEMO_BG);
    gc_render_board(env);

    if (ui->paused) gc_demo_render_paused(client);

    // One condensed tally line: "#4  G (L3) 3 WINS (30%)  C (L4) 9 WINS (70%)"
    GcDemoTally* tally = &ui->tally;
    float games = tally->games > 0 ? (float)tally->games : 1.0f;
    int bar_y = GC_BOARD_H * cell;
    int x = 12;
    const char* seg = TextFormat("#%d", tally->games + 1);
    DrawText(seg, x, bar_y + 16, 16, RAYWHITE);
    x += MeasureText(seg, 16) + 14;
    seg = TextFormat("G (L%s) %d WINS (%.0f%%)",
        gc_demo_levels[ui->ai_level[GC_GUERRILLA]].label, tally->guerrilla_wins,
        100.0f * (float)tally->guerrilla_wins / games);
    DrawText(seg, x, bar_y + 16, 16, GC_DEMO_GUERRILLA);
    x += MeasureText(seg, 16) + 14;
    seg = TextFormat("C (L%s) %d WINS (%.0f%%)",
        gc_demo_levels[ui->ai_level[GC_COIN]].label, tally->coin_wins,
        100.0f * (float)tally->coin_wins / games);
    DrawText(seg, x, bar_y + 16, 16, GC_DEMO_COIN);
    x += MeasureText(seg, 16) + 14;

    const char* counts = TextFormat("G %d", GC_MAX_GUERRILLAS - env->guerrilla_count);
    int counts_x = client->width - 96 - MeasureText(counts, 16);
    if (counts_x > x) {
        DrawText(counts, counts_x, bar_y + 16, 16, GC_DEMO_GUERRILLA);
    }
    int menu_clicked = gc_demo_button(
        (Rectangle){(float)client->width - 84.0f, (float)bar_y + 8.0f, 72, 32}, "MENU", 0);
    EndDrawing();
    return menu_clicked;
}

static int gc_demo_bot_action(GuerrillaCheckers* env, GcDemoUi* ui) {
    const GcDemoLevel* level = &gc_demo_levels[ui->ai_level[env->player_to_move]];
    if (level->opponent == GC_DEMO_NET_BOT) {
        if (gc_demo_net_loaded) return gc_demo_net_action(env);
        level = &gc_demo_levels[GC_DEMO_LEVEL_MCTS_2K];
    }
    env->opponent = level->opponent;
    env->mcts_iterations = level->mcts_iterations;
    return gc_bot_action(env);
}

static void demo(void) {
    // The client applies actions and renders directly (menus, hints, no
    // auto-reset); these header entry points are for binding.c.
    (void)c_step;
    (void)c_render;

    GuerrillaCheckers env = {0};
    env.num_agents = 1;
    env.max_episode_length = 256;
    env.render_fps = 60;
    env.selfplay = 1;  // the client drives both sides turn by turn
    env.side_cfg = 0;
    env.opponent = GC_BOT_GREEDY;
    env.mcts_iterations = 2000;
    env.mcts_exploration = GC_MCTS_DEFAULT_EXPLORATION;
    env.mcts_rollout = GC_MCTS_ROLLOUT_GREEDY;
    env.rng = 1234u;

    gc_demo_allocate(&env);
    c_reset(&env);

    gc_demo_net_loaded =
        gc_demo_net_init(&gc_demo_nets[GC_GUERRILLA], GC_DEMO_NET_WEIGHTS) &&
        gc_demo_net_init(&gc_demo_nets[GC_COIN], GC_DEMO_NET_WEIGHTS);

    env.client = gc_make_client(&env);
    SetExitKey(KEY_NULL);  // ESC navigates to the menu instead of quitting

    GcDemoUi ui = {0};
    ui.mode = GC_UI_MENU;
    ui.ctrl[GC_GUERRILLA] = GC_CTRL_HUMAN;
    ui.ctrl[GC_COIN] = GC_CTRL_AI;
    ui.ai_level[GC_GUERRILLA] = gc_demo_net_loaded ?
        GC_DEMO_LEVEL_NET : GC_DEMO_LEVEL_MCTS_2K;
    ui.ai_level[GC_COIN] = ui.ai_level[GC_GUERRILLA];
    ui.selected = GC_DEMO_NOOP;

    while (!WindowShouldClose()) {
        if (ui.mode == GC_UI_MENU) {
            gc_demo_render_menu(&env, &ui);
            continue;
        }

        if (IsKeyPressed(KEY_ESCAPE)) {
            ui.mode = GC_UI_MENU;
            gc_demo_render_menu(&env, &ui);
            continue;
        }

        if (IsKeyPressed(KEY_R)) {
            memset(&ui.tally, 0, sizeof(ui.tally));
            gc_demo_start_game(&env, &ui);
        }

        if (ui.mode == GC_UI_TOURNEY) {
            if (IsKeyPressed(KEY_SPACE)) ui.paused = !ui.paused;
            // Simulate as many moves as fit in the frame budget, then render
            // the current position and the running tally.
            double frame_end = GetTime() + GC_DEMO_TOURNEY_BUDGET;
            while (!ui.paused && GetTime() < frame_end) {
                if (env.game_over) {
                    ui.tally.games++;
                    if (env.winner == GC_GUERRILLA) ui.tally.guerrilla_wins++;
                    else ui.tally.coin_wins++;
                    env.rng ^= (unsigned int)(GetTime() * 1e6) | 1u;
                    c_reset(&env);
                    gc_demo_net_reset();
                    continue;
                }
                gc_demo_apply(&env, &ui, gc_demo_bot_action(&env, &ui));
            }
            if (gc_demo_render_tourney(&env, &ui)) ui.mode = GC_UI_MENU;
            continue;
        }

        if (env.game_over) {
            if (ui.over_wait > 0) {
                ui.over_wait--;
            } else if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                ui.mode = GC_UI_MENU;
                continue;  // consume the click before the menu renders
            }
            if (gc_demo_render_play(&env, &ui)) ui.mode = GC_UI_MENU;
            continue;
        }

        // AI vs AI games can be paused like tournaments.
        if (ui.ctrl[GC_GUERRILLA] == GC_CTRL_AI && ui.ctrl[GC_COIN] == GC_CTRL_AI &&
                IsKeyPressed(KEY_SPACE)) {
            ui.paused = !ui.paused;
        }

        int action = GC_DEMO_NOOP;
        if (ui.ctrl[env.player_to_move] == GC_CTRL_HUMAN) {
            int prev_selected = ui.selected;
            action = gc_demo_human_action(&env, &ui.selected);
            // The trail marks the opponent's last move; drop it as soon as
            // the player starts their own.
            if (ui.selected != prev_selected) {
                memset(&ui.trail, 0, sizeof(ui.trail));
            }
        } else if (ui.paused) {
            // Hold the position until space is pressed again.
        } else if (ui.ai_wait > 0) {
            // Keep presenting "THINKING..." frames before the (blocking) search
            // so the status stays visible while MCTS runs.
            ui.ai_wait--;
        } else {
            action = gc_demo_bot_action(&env, &ui);
        }

        if (action != GC_DEMO_NOOP) {
            gc_demo_apply(&env, &ui, action);
            ui.selected = GC_DEMO_NOOP;
            ui.ai_wait = GC_DEMO_AI_WAIT;
            if (env.game_over) {
                ui.over_wait = GC_DEMO_OVER_WAIT;
            } else if (env.coin_must_capture &&
                    ui.ctrl[GC_COIN] == GC_CTRL_HUMAN) {
                // Capture chains continue with the same piece: keep it selected.
                ui.selected = env.coin_previous_cell;
            }
        }

        if (gc_demo_render_play(&env, &ui)) ui.mode = GC_UI_MENU;
    }

    gc_demo_free(&env);
}

int main(void) {
    demo();
    return 0;
}
