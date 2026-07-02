#pragma once

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

// Guerrilla Checkers is an asymmetric hybrid of Checkers and Go.
// https://mischa-u.itch.io/guerrilla-checkers
// https://brtrain.wordpress.com/tag/guerrilla-checkers/
// https://nestorgames.com/rulebooks/GUERRILLACHECKERS_EN.pdf

#define GC_BOARD_W 8
#define GC_BOARD_H 8
#define GC_COIN_CELLS (GC_BOARD_W * GC_BOARD_H)
#define GC_G_W (GC_BOARD_W - 1)
#define GC_G_H (GC_BOARD_H - 1)
#define GC_G_CELLS (GC_G_W * GC_G_H)
#define GC_MAX_GUERRILLAS 66
#define GC_ACTIONS 256
#define GC_G_ACTIONS (GC_G_CELLS * 4)
#define GC_OBS_SIZE (GC_G_CELLS + GC_COIN_CELLS + 7)
#define GC_INVALID_ACTION_REWARD -1.0f

enum {
    GC_NONE = 0,
    GC_GUERRILLA = 1,
    GC_COIN = 2,
};

// opponent kwarg: which bot plays the non-agent side when selfplay == 0.
enum {
    GC_BOT_RANDOM = 0,
    GC_BOT_GREEDY = 1,
    GC_BOT_MCTS = 2,
};

typedef struct Log {
    float perf;   // scored-side win rate; higher is better
    float score;  // scored-side terminal reward
    float episode_return;
    float episode_length;
    float invalid_rate;  // mean fraction of a side's decisions that were illegal
    float games_as_guerrilla;
    float wins_as_guerrilla;
    float games_as_coin;
    float wins_as_coin;
    float n;
} Log;

typedef struct Client Client;

typedef struct GuerrillaCheckers {
    uint8_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    Log log;
    Client* client;
    unsigned int rng;

    uint8_t coin_cells[GC_COIN_CELLS];
    uint8_t guerrilla_cells[GC_G_CELLS];
    int player_to_move;
    int coin_must_capture;
    int coin_previous_cell;
    int guerrilla_previous_cell;
    int guerrilla_cells_count;
    int guerrilla_count;
    int guerrilla_placed_this_turn;
    int game_over;
    int winner;
    int tick;
    int legal_count;
    int invalid_this_episode;
    int max_episode_length;
    int render_fps;

    // selfplay == 1: one policy controls both sides on alternating turns
    // (reward is actor-relative). selfplay == 0: the policy plays agent_side
    // and a built-in bot (opponent) plays the other side inside c_step, so the
    // reward is always from the agent's fixed perspective.
    int selfplay;
    int side_cfg;    // configured agent side: 0 = random, else GC_GUERRILLA / GC_COIN
    int agent_side;  // resolved agent side for the current episode
    int opponent;    // GC_BOT_RANDOM / GC_BOT_GREEDY / GC_BOT_MCTS
    int mcts_iterations;      // MCTS search budget per move (opponent == GC_BOT_MCTS)
    float mcts_exploration;   // UCB1 exploration constant
    int mcts_rollout;         // GC_MCTS_ROLLOUT_RANDOM / GC_MCTS_ROLLOUT_GREEDY
} GuerrillaCheckers;

static inline unsigned int gc_rand(GuerrillaCheckers* env) {
    return (unsigned int)rand_r(&env->rng);
}

static inline int gc_coin_pos(int x, int y) {
    return y * GC_BOARD_W + x;
}

static inline int gc_coin_x(int pos) {
    return pos % GC_BOARD_W;
}

static inline int gc_coin_y(int pos) {
    return pos / GC_BOARD_W;
}

static inline int gc_g_pos(int x, int y) {
    return y * GC_G_W + x;
}

static inline int gc_g_x(int pos) {
    return pos % GC_G_W;
}

static inline int gc_g_y(int pos) {
    return pos / GC_G_W;
}

static inline int gc_valid_coin_xy(int x, int y) {
    return x >= 0 && x < GC_BOARD_W && y >= 0 && y < GC_BOARD_H;
}

static inline int gc_valid_g_xy(int x, int y) {
    return x >= 0 && x < GC_G_W && y >= 0 && y < GC_G_H;
}

static inline int gc_valid_g_pos(int pos) {
    return pos >= 0 && pos < GC_G_CELLS;
}

static int gc_coin_jump_cell(int src, int dst) {
    int sx = gc_coin_x(src);
    int sy = gc_coin_y(src);
    int dx = gc_coin_x(dst) - sx;
    int dy = gc_coin_y(dst) - sy;
    int gx = sx + (dx < 0 ? -1 : 0);
    int gy = sy + (dy < 0 ? -1 : 0);
    return gc_g_pos(gx, gy);
}

static int gc_coin_neighbor(int src, int dir) {
    static const int dx[4] = {-1, 1, -1, 1};
    static const int dy[4] = {-1, -1, 1, 1};
    int x = gc_coin_x(src) + dx[dir];
    int y = gc_coin_y(src) + dy[dir];
    if (!gc_valid_coin_xy(x, y)) return -1;
    return gc_coin_pos(x, y);
}

static int gc_g_neighbor(int src, int dir) {
    static const int dx[4] = {-1, 0, 0, 1};
    static const int dy[4] = {0, -1, 1, 0};
    int x = gc_g_x(src) + dx[dir];
    int y = gc_g_y(src) + dy[dir];
    if (!gc_valid_g_xy(x, y)) return -1;
    return gc_g_pos(x, y);
}

static int gc_adjacent_to_existing_guerrilla(GuerrillaCheckers* env, int pos) {
    for (int dir = 0; dir < 4; dir++) {
        int n = gc_g_neighbor(pos, dir);
        if (n >= 0 && env->guerrilla_cells[n]) return 1;
    }
    return 0;
}

static int gc_adjacent_to_previous_guerrilla(GuerrillaCheckers* env, int pos) {
    if (env->guerrilla_previous_cell < 0) return 0;
    for (int dir = 0; dir < 4; dir++) {
        if (gc_g_neighbor(env->guerrilla_previous_cell, dir) == pos) return 1;
    }
    return 0;
}

static int gc_can_place_guerrilla(GuerrillaCheckers* env, int dest) {
    if (!gc_valid_g_pos(dest)) return 0;
    if (env->guerrilla_count == 0) return 1;
    if (env->guerrilla_cells[dest]) return 0;
    if (env->guerrilla_placed_this_turn == 0) {
        return gc_adjacent_to_existing_guerrilla(env, dest);
    }
    return gc_adjacent_to_previous_guerrilla(env, dest);
}

static int gc_coin_can_capture(GuerrillaCheckers* env, int src) {
    for (int dir = 0; dir < 4; dir++) {
        int dst = gc_coin_neighbor(src, dir);
        if (dst < 0 || env->coin_cells[dst]) continue;
        int jump_cell = gc_coin_jump_cell(src, dst);
        if (gc_valid_g_pos(jump_cell) && env->guerrilla_cells[jump_cell]) {
            return 1;
        }
    }
    return 0;
}

static int gc_can_move_coin(GuerrillaCheckers* env, int src, int dst) {
    if (src < 0 || src >= GC_COIN_CELLS || dst < 0 || dst >= GC_COIN_CELLS) return 0;
    if (!env->coin_cells[src] || env->coin_cells[dst]) return 0;

    int dx = abs(gc_coin_x(dst) - gc_coin_x(src));
    int dy = abs(gc_coin_y(dst) - gc_coin_y(src));
    if (dx != 1 || dy != 1) return 0;

    int jump_cell = gc_coin_jump_cell(src, dst);
    if (!gc_valid_g_pos(jump_cell)) return 0;
    if (env->coin_must_capture) {
        if (src != env->coin_previous_cell) return 0;
        if (!env->guerrilla_cells[jump_cell]) return 0;
    }
    return 1;
}

static int gc_guerrilla_has_second_placement(GuerrillaCheckers* env) {
    if (env->player_to_move != GC_GUERRILLA || env->guerrilla_placed_this_turn != 1) {
        return 0;
    }
    for (int dir = 0; dir < 4; dir++) {
        int n = gc_g_neighbor(env->guerrilla_previous_cell, dir);
        if (n >= 0 && !env->guerrilla_cells[n]) return 1;
    }
    return 0;
}

static void gc_check_victory(GuerrillaCheckers* env) {
    if (env->game_over) return;

    int coin_count = 0;
    for (int i = 0; i < GC_COIN_CELLS; i++) {
        coin_count += env->coin_cells[i] ? 1 : 0;
    }

    if (coin_count == 0) {
        env->game_over = 1;
        env->winner = GC_GUERRILLA;
    } else if (env->guerrilla_count == GC_MAX_GUERRILLAS ||
            (env->guerrilla_count > 0 && env->guerrilla_cells_count == 0)) {
        env->game_over = 1;
        env->winner = GC_COIN;
    } else if (env->player_to_move == GC_GUERRILLA && env->guerrilla_placed_this_turn == 1) {
        // Call unconditionally so the function is referenced in release builds
        // (assert() is compiled out under -DNDEBUG); (void) silences the unused
        // result there.
        int has_second = gc_guerrilla_has_second_placement(env);
        assert(has_second &&
            "Guerrilla Checkers reached a non-terminal state with no legal second placement");
        (void)has_second;
    }
}

static void gc_check_timeout(GuerrillaCheckers* env) {
    if (env->game_over) return;
    if (env->tick >= env->max_episode_length) {
        // Undecided at the step limit. The Guerrilla's only win condition is to
        // clear the board, so failing to do so in time is a Guerrilla loss (the
        // same outcome as running out of pieces). Deciding by side rather than by
        // whoever happens to be on-move keeps timeouts from always falling on the
        // learner in bot mode (where player_to_move is always agent_side here).
        env->game_over = 1;
        env->winner = GC_COIN;
    }
}

static int gc_check_guerrilla_capture(GuerrillaCheckers* env) {
    int captured = 0;
    for (int c = 0; c < GC_COIN_CELLS; c++) {
        if (!env->coin_cells[c]) continue;

        int x = gc_coin_x(c);
        int y = gc_coin_y(c);
        int enemy_count = 0;
        int surrounded = 1;
        for (int ox = -1; ox <= 0; ox++) {
            for (int oy = -1; oy <= 0; oy++) {
                int gx = x + ox;
                int gy = y + oy;
                if (!gc_valid_g_xy(gx, gy)) continue;
                enemy_count++;
                if (!env->guerrilla_cells[gc_g_pos(gx, gy)]) surrounded = 0;
            }
        }
        if (enemy_count > 0 && surrounded) {
            env->coin_cells[c] = 0;
            captured++;
        }
    }
    return captured;
}

static int gc_place_guerrilla(GuerrillaCheckers* env, int dest) {
    if (!gc_can_place_guerrilla(env, dest)) return -1;

    env->guerrilla_cells[dest] = 1;
    env->guerrilla_cells_count++;
    env->guerrilla_count++;
    env->guerrilla_placed_this_turn++;
    env->guerrilla_previous_cell = dest;

    int captures = gc_check_guerrilla_capture(env);
    gc_check_victory(env);

    if (!env->game_over && env->guerrilla_placed_this_turn == 2) {
        env->guerrilla_placed_this_turn = 0;
        env->guerrilla_previous_cell = -1;
        env->player_to_move = GC_COIN;
        gc_check_victory(env);
    }
    return captures;
}

static int gc_move_coin(GuerrillaCheckers* env, int src, int dst) {
    if (!gc_can_move_coin(env, src, dst)) return -1;

    int jump_cell = gc_coin_jump_cell(src, dst);
    int captured = env->guerrilla_cells[jump_cell] ? 1 : 0;

    env->coin_cells[src] = 0;
    env->coin_cells[dst] = 1;

    if (captured) {
        env->guerrilla_cells[jump_cell] = 0;
        env->guerrilla_cells_count--;
        if (env->guerrilla_cells_count == 0) {
            env->game_over = 1;
            env->winner = GC_COIN;
            return 1;
        }

        if (gc_coin_can_capture(env, dst)) {
            env->coin_must_capture = 1;
            env->coin_previous_cell = dst;
            gc_check_victory(env);
            return 1;
        }
    }

    env->coin_must_capture = 0;
    env->coin_previous_cell = -1;
    env->player_to_move = GC_GUERRILLA;
    gc_check_victory(env);
    return captured;
}

static int gc_decode_guerrilla_action(int action, int* first, int* second) {
    if (action < 0 || action >= GC_G_ACTIONS) return 0;
    *first = action / 4;
    static const int dirs[4] = {2, 3, 0, 1};  // down/right first so action 0 is valid at reset
    *second = gc_g_neighbor(*first, dirs[action % 4]);
    return 1;
}

static int gc_action_is_legal(GuerrillaCheckers* env, int action) {
    if (env->game_over) return 0;
    if (action < 0 || action >= GC_ACTIONS) return 0;

    if (env->player_to_move == GC_GUERRILLA) {
        int first;
        int second;
        if (!gc_decode_guerrilla_action(action, &first, &second)) return 0;
        if (!gc_can_place_guerrilla(env, first)) return 0;

        if (second < 0) return 0;
        if (!gc_valid_g_pos(second) || env->guerrilla_cells[second]) return 0;
        return 1;
    }

    int src = action / 4;
    int dir = action % 4;
    int dst = gc_coin_neighbor(src, dir);
    return gc_can_move_coin(env, src, dst);
}

static int gc_enumerate_legal(GuerrillaCheckers* env, int* out) {
    int n = 0;
    if (env->game_over) return 0;

    if (env->player_to_move == GC_GUERRILLA) {
        static const int dirs[4] = {2, 3, 0, 1};
        for (int first = 0; first < GC_G_CELLS; first++) {
            if (!gc_can_place_guerrilla(env, first)) continue;
            for (int action_dir = 0; action_dir < 4; action_dir++) {
                int second = gc_g_neighbor(first, dirs[action_dir]);
                if (second < 0 || env->guerrilla_cells[second]) continue;
                out[n++] = first * 4 + action_dir;
            }
        }
    } else {
        for (int src = 0; src < GC_COIN_CELLS; src++) {
            if (!env->coin_cells[src]) continue;
            for (int dir = 0; dir < 4; dir++) {
                int dst = gc_coin_neighbor(src, dir);
                if (!gc_can_move_coin(env, src, dst)) continue;
                out[n++] = src * 4 + dir;
            }
        }
    }
    return n;
}

static int gc_rebuild_action_mask(GuerrillaCheckers* env) {
    if (env->action_mask != NULL) {
        memset(env->action_mask, 0, GC_ACTIONS * sizeof(unsigned char));
    }

    int legal[GC_ACTIONS];
    int count = gc_enumerate_legal(env, legal);
    if (env->action_mask != NULL) {
        for (int i = 0; i < count; i++) {
            env->action_mask[legal[i]] = 1;
        }
    }

    env->legal_count = count;
    return count;
}

static void gc_apply_no_legal_loss(GuerrillaCheckers* env, int legal_count) {
    if (!env->game_over && legal_count == 0) {
        env->game_over = 1;
        env->winner = env->player_to_move == GC_GUERRILLA ? GC_COIN : GC_GUERRILLA;
    }
}

static int gc_prepare_turn(GuerrillaCheckers* env) {
    int legal_count = gc_rebuild_action_mask(env);
    gc_apply_no_legal_loss(env, legal_count);
    if (env->game_over) gc_rebuild_action_mask(env);
    return !env->game_over;
}

static int gc_action_allowed_by_current_mask(GuerrillaCheckers* env, int action) {
    if (action < 0 || action >= GC_ACTIONS) return 0;
    if (env->action_mask != NULL) return env->action_mask[action] != 0;
    return gc_action_is_legal(env, action);
}

// Apply one legal action for the current player; returns the number of enemy
// pieces captured. Shared by the learner and the built-in bot.
static int gc_apply_action(GuerrillaCheckers* env, int action) {
    if (env->player_to_move == GC_GUERRILLA) {
        int first;
        int second;
        gc_decode_guerrilla_action(action, &first, &second);
        int captures = gc_place_guerrilla(env, first);
        if (!env->game_over && second >= 0) {
            int second_captures = gc_place_guerrilla(env, second);
            if (second_captures > 0) captures += second_captures;
        }
        return captures > 0 ? captures : 0;
    }
    int src = action / 4;
    int dst = gc_coin_neighbor(src, action % 4);
    int captures = gc_move_coin(env, src, dst);
    return captures > 0 ? captures : 0;
}

static int gc_collect_legal(GuerrillaCheckers* env, int* out) {
    int n = 0;
    if (env->action_mask != NULL) {
        for (int action = 0; action < GC_ACTIONS; action++) {
            if (env->action_mask[action]) out[n++] = action;
        }
        return n;
    }

    return gc_enumerate_legal(env, out);
}

static inline void gc_add_capture_candidates(int pos, int* candidates, int* n) {
    if (!gc_valid_g_pos(pos)) return;
    int gx = gc_g_x(pos);
    int gy = gc_g_y(pos);
    for (int dx = 0; dx <= 1; dx++) {
        for (int dy = 0; dy <= 1; dy++) {
            int cx = gx + dx;
            int cy = gy + dy;
            if (!gc_valid_coin_xy(cx, cy)) continue;
            int coin = gc_coin_pos(cx, cy);
            int seen = 0;
            for (int i = 0; i < *n; i++) {
                if (candidates[i] == coin) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) candidates[(*n)++] = coin;
        }
    }
}

// Immediate captures a candidate move would make, without mutating the board
// (used to rank moves for the greedy bot).
static int gc_action_capture_score(GuerrillaCheckers* env, int action) {
    if (env->player_to_move == GC_GUERRILLA) {
        int first;
        int second;
        gc_decode_guerrilla_action(action, &first, &second);
        int candidates[8];
        int candidate_count = 0;
        gc_add_capture_candidates(first, candidates, &candidate_count);
        gc_add_capture_candidates(second, candidates, &candidate_count);
        int count = 0;
        // Captures are applied eagerly after every Guerrilla placement, so a
        // reachable board cannot already contain a fully surrounded coin. Only
        // coins whose corner set includes one of the newly placed stones can
        // become newly surrounded by this action.
        for (int i = 0; i < candidate_count; i++) {
            int c = candidates[i];
            if (!env->coin_cells[c]) continue;
            int x = gc_coin_x(c);
            int y = gc_coin_y(c);
            int enemy_count = 0;
            int surrounded = 1;
            for (int ox = -1; ox <= 0; ox++) {
                for (int oy = -1; oy <= 0; oy++) {
                    int gx = x + ox;
                    int gy = y + oy;
                    if (!gc_valid_g_xy(gx, gy)) continue;
                    enemy_count++;
                    int gp = gc_g_pos(gx, gy);
                    int occupied = env->guerrilla_cells[gp] || gp == first ||
                        (second >= 0 && gp == second);
                    if (!occupied) surrounded = 0;
                }
            }
            if (enemy_count > 0 && surrounded) count++;
        }
        return count;
    }
    int src = action / 4;
    int dst = gc_coin_neighbor(src, action % 4);
    if (dst < 0) return 0;
    int jump_cell = gc_coin_jump_cell(src, dst);
    return gc_valid_g_pos(jump_cell) && env->guerrilla_cells[jump_cell] ? 1 : 0;
}

static int gc_greedy_pick(GuerrillaCheckers* rng_env, GuerrillaCheckers* state,
        const int* legal, int n) {
    int best[GC_ACTIONS];
    int best_count = 0;
    int best_score = -1;
    for (int i = 0; i < n; i++) {
        int score = gc_action_capture_score(state, legal[i]);
        if (score > best_score) {
            best_score = score;
            best_count = 0;
            best[best_count++] = legal[i];
        } else if (score == best_score) {
            best[best_count++] = legal[i];
        }
    }
    return best[gc_rand(rng_env) % (unsigned int)best_count];
}

// MCTS opponent (opponent == GC_BOT_MCTS). Depends on the game primitives above.
#include "mcts.h"

// Pick a move for the built-in opponent.
static int gc_bot_action(GuerrillaCheckers* env) {
    if (env->opponent == GC_BOT_MCTS) return gc_mcts_action(env);

    int legal[GC_ACTIONS];
    int n = gc_collect_legal(env, legal);
    assert(n > 0 &&
        "Guerrilla Checkers bot reached a non-terminal state with no legal moves");
    if (env->opponent != GC_BOT_GREEDY) {
        return legal[gc_rand(env) % (unsigned int)n];
    }
    return gc_greedy_pick(env, env, legal, n);
}

static void gc_play_bot_turns(GuerrillaCheckers* env) {
    while (!env->game_over && env->player_to_move != env->agent_side) {
        if (!gc_prepare_turn(env)) return;
        if (env->player_to_move == env->agent_side) return;
        int action = gc_bot_action(env);
        gc_apply_action(env, action);
    }
}

static void gc_compute_observations(GuerrillaCheckers* env) {
    int idx = 0;
    for (int i = 0; i < GC_G_CELLS; i++) {
        env->observations[idx++] = env->guerrilla_cells[i] ? 1 : 0;
    }
    for (int i = 0; i < GC_COIN_CELLS; i++) {
        env->observations[idx++] = env->coin_cells[i] ? 1 : 0;
    }

    // ByteTensor observations are cast to float by the default encoder. Keep the
    // representation compact and byte-native: 0 is "none/false", ids are +1.
    env->observations[idx++] = env->player_to_move == GC_GUERRILLA ? 1 : 2;
    env->observations[idx++] = env->coin_must_capture ? 1 : 0;
    env->observations[idx++] = env->coin_previous_cell < 0 ? 0 :
        (uint8_t)(env->coin_previous_cell + 1);
    env->observations[idx++] = env->guerrilla_previous_cell < 0 ? 0 :
        (uint8_t)(env->guerrilla_previous_cell + 1);
    env->observations[idx++] = (uint8_t)env->guerrilla_placed_this_turn;
    env->observations[idx++] = (uint8_t)env->guerrilla_count;
    env->observations[idx++] = (uint8_t)env->guerrilla_cells_count;
    assert(idx == GC_OBS_SIZE);
}

// score_side is the perspective the terminal reward and per-side metrics are
// credited to: the actor in selfplay, otherwise the (fixed) agent side.
static void gc_finish_step(GuerrillaCheckers* env, int score_side, float reward) {
    gc_prepare_turn(env);

    if (env->game_over) {
        reward = env->winner == score_side ? 1.0f : -1.0f;
        env->terminals[0] = 1.0f;

        float guerrilla_win = env->winner == GC_GUERRILLA ? 1.0f : 0.0f;
        float coin_win = env->winner == GC_COIN ? 1.0f : 0.0f;
        // In selfplay one policy plays both sides, so every game counts toward
        // both sides; in bot mode the game counts toward the agent's side only.
        if (env->selfplay || score_side == GC_GUERRILLA) {
            env->log.games_as_guerrilla += 1.0f;
            env->log.wins_as_guerrilla += guerrilla_win;
        }
        if (env->selfplay || score_side == GC_COIN) {
            env->log.games_as_coin += 1.0f;
            env->log.wins_as_coin += coin_win;
        }
        // perf follows the Ocean convention: scored-side win rate. The side
        // counters above expose Guerrilla/COIN balance separately.
        env->log.perf += env->winner == score_side ? 1.0f : 0.0f;
        // Bounded [0,1]: fraction of this episode's decisions that were illegal.
        env->log.invalid_rate += (float)env->invalid_this_episode /
            (float)(env->tick > 0 ? env->tick : 1);
        env->log.score += reward;
        env->log.episode_length += (float)env->tick;
        env->log.n += 1.0f;
    }

    env->rewards[0] = fmaxf(-1.0f, fminf(1.0f, reward));
    env->log.episode_return += env->rewards[0];
    gc_compute_observations(env);
}

static void c_reset(GuerrillaCheckers* env) {
    memset(env->coin_cells, 0, sizeof(env->coin_cells));
    memset(env->guerrilla_cells, 0, sizeof(env->guerrilla_cells));

    static const int starts[6][2] = {
        {3, 2}, {2, 3}, {4, 3}, {3, 4}, {5, 4}, {4, 5},
    };
    for (int i = 0; i < 6; i++) {
        env->coin_cells[gc_coin_pos(starts[i][0], starts[i][1])] = 1;
    }

    env->player_to_move = GC_GUERRILLA;
    env->coin_must_capture = 0;
    env->coin_previous_cell = -1;
    env->guerrilla_previous_cell = -1;
    env->guerrilla_cells_count = 0;
    env->guerrilla_count = 0;
    env->guerrilla_placed_this_turn = 0;
    env->game_over = 0;
    env->winner = GC_NONE;
    env->tick = 0;
    env->legal_count = 0;
    env->invalid_this_episode = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    if (env->selfplay) {
        env->agent_side = GC_GUERRILLA;  // unused for scoring; the actor is used
    } else if (env->side_cfg == GC_GUERRILLA || env->side_cfg == GC_COIN) {
        env->agent_side = env->side_cfg;
    } else {
        env->agent_side = (gc_rand(env) & 1u) ? GC_GUERRILLA : GC_COIN;
    }
    // When the agent plays COIN, let the Guerrilla bot open so the first
    // observation is on the agent's turn.
    if (!env->selfplay) {
        gc_play_bot_turns(env);
    }

    gc_prepare_turn(env);
    gc_compute_observations(env);
}

static void gc_reset_after_terminal_step(GuerrillaCheckers* env) {
    if (env->terminals[0] != 1.0f) return;
    float reward = env->rewards[0];
    c_reset(env);
    env->rewards[0] = reward;
    env->terminals[0] = 1.0f;
}

static void c_step(GuerrillaCheckers* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int actor = env->player_to_move;
    int action = (int)env->actions[0];
    assert(env->legal_count > 0 &&
        "Guerrilla Checkers step reached a non-terminal state with no legal moves");
    int legal = gc_action_allowed_by_current_mask(env, action);
    env->tick++;

    // Reward and metrics use the actor in selfplay, the fixed agent side vs bot.
    int score_side = env->selfplay ? actor : env->agent_side;

    if (!legal) {
        // All rollout paths (CUDA, and CPU/MPS via sample_logits) now consume
        // MY_ACTION_MASK before sampling, so a masked policy never lands here and
        // invalid_rate stays ~0. As a safety net for unmasked/eval use, treat an
        // illegal action as a negative no-op (not an instant forfeit) so stalling
        // never beats legal play; the timeout below bounds any stall.
        env->invalid_this_episode++;
        gc_check_timeout(env);
        gc_finish_step(env, score_side, GC_INVALID_ACTION_REWARD);
        gc_reset_after_terminal_step(env);
        return;
    }

    int captures = gc_apply_action(env, action);
    float reward = 0.0f;
    if (captures > 0) {
        reward += (actor == GC_GUERRILLA ? 0.05f : 0.03f) * (float)captures;
    }

    // Bot mode: play the opponent's reply(ies) so the game returns to the
    // agent's turn (or ends) before we score this step.
    if (!env->selfplay) {
        gc_play_bot_turns(env);
    }

    gc_check_timeout(env);
    gc_finish_step(env, score_side, reward);
    gc_reset_after_terminal_step(env);
}

static void c_close(GuerrillaCheckers* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

struct Client {
    int width;
    int height;
    int cell;
};

static Client* gc_make_client(GuerrillaCheckers* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->cell = 72;
    client->width = GC_BOARD_W * client->cell;
    client->height = GC_BOARD_H * client->cell + 48;
    InitWindow(client->width, client->height, "PufferLib Guerrilla Checkers");
    SetTargetFPS(env->render_fps);
    return client;
}

// Board squares and pieces only; callers wrap this in Begin/EndDrawing and
// draw their own status text (and, for the standalone client, move hints).
static void gc_render_board(GuerrillaCheckers* env) {
    int cell = env->client->cell;

    for (int y = 0; y < GC_BOARD_H; y++) {
        for (int x = 0; x < GC_BOARD_W; x++) {
            Color square = ((x + y) & 1) ? (Color){54, 70, 78, 255} : (Color){36, 48, 54, 255};
            DrawRectangle(x * cell, y * cell, cell, cell, square);
            DrawRectangleLines(x * cell, y * cell, cell, cell, (Color){15, 20, 24, 255});
            int pos = gc_coin_pos(x, y);
            if (env->coin_cells[pos]) {
                DrawCircle(x * cell + cell / 2, y * cell + cell / 2, cell * 0.28f,
                    (Color){232, 198, 83, 255});
                DrawCircleLines(x * cell + cell / 2, y * cell + cell / 2, cell * 0.28f,
                    (Color){96, 74, 22, 255});
            }
        }
    }

    for (int y = 0; y < GC_G_H; y++) {
        for (int x = 0; x < GC_G_W; x++) {
            int pos = gc_g_pos(x, y);
            int cx = x * cell + cell;
            int cy = y * cell + cell;
            DrawCircle(cx, cy, 5.0f, (Color){104, 126, 132, 255});
            if (env->guerrilla_cells[pos]) {
                DrawCircle(cx, cy, cell * 0.20f, (Color){206, 72, 72, 255});
                DrawCircleLines(cx, cy, cell * 0.20f, (Color){88, 28, 28, 255});
            }
        }
    }
}

static void c_render(GuerrillaCheckers* env) {
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (env->client == NULL) env->client = gc_make_client(env);

    int cell = env->client->cell;
    BeginDrawing();
    ClearBackground((Color){18, 24, 28, 255});
    gc_render_board(env);

    const char* side = env->player_to_move == GC_GUERRILLA ? "Guerrilla" : "COIN";
    const char* status = env->game_over ?
        (env->winner == GC_GUERRILLA ? "Guerrilla wins" :
         "COIN wins") : side;
    DrawText(status, 12, GC_BOARD_H * cell + 14, 22, RAYWHITE);
    DrawText(TextFormat("turn %d  guerrillas %d/%d", env->tick,
        env->guerrilla_count, GC_MAX_GUERRILLAS),
        250, GC_BOARD_H * cell + 16, 18, (Color){190, 204, 208, 255});
    EndDrawing();
}
