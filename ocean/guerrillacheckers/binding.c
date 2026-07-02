#include "guerrillacheckers.h"

#define OBS_SIZE GC_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {GC_ACTIONS}
#define OBS_TENSOR_T ByteTensor
#define MY_ACTION_MASK GC_ACTIONS

#define Env GuerrillaCheckers
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_episode_length = (int)dict_get(kwargs, "max_episode_length")->value;
    env->render_fps = (int)dict_get(kwargs, "render_fps")->value;
    env->selfplay = (int)dict_get(kwargs, "selfplay")->value;
    env->side_cfg = (int)dict_get(kwargs, "side")->value;
    env->opponent = (int)dict_get(kwargs, "opponent")->value;
    env->mcts_iterations = (int)dict_get(kwargs, "mcts_iterations")->value;
    env->mcts_exploration = (float)dict_get(kwargs, "mcts_exploration")->value;
    env->mcts_rollout = (int)dict_get(kwargs, "mcts_rollout")->value;
    // Default vec initialization stores the per-env index in rng before my_init.
    // Scramble it so env index 0 still gets a non-degenerate seed.
    env->rng = env->rng * 2654435761u + 12345u;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "invalid_rate", log->invalid_rate);
    dict_set(out, "games_as_guerrilla", log->games_as_guerrilla);
    dict_set(out, "wins_as_guerrilla", log->wins_as_guerrilla);
    dict_set(out, "games_as_coin", log->games_as_coin);
    dict_set(out, "wins_as_coin", log->wins_as_coin);
}
