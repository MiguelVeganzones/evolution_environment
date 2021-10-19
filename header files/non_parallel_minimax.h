#pragma once
#include <iostream>
#include <map>
#include "board.h"
#include "parallel_hashmap/phmap.h"
#include "c4_brain.h"

extern const uint_fast8_t np_sdepth; //calls to minimax  <= sx + sx**2 + .... + sx**sdepth (creo)
//phmap::parallel_flat_hash_map

extern thread_local phmap::parallel_flat_hash_map<std::pair<uint64_t, uint64_t>, uint_fast8_t> np_computed_boards;
extern thread_local phmap::parallel_flat_hash_map<std::pair<uint64_t, uint64_t>, uint_fast8_t> np_seen_boards; // board num and move chosen

uint_fast8_t heuristic(const board& b);

uint_fast8_t np_c4_minimax(const board& prev_board, const _c4_brain::c4_brain* brain, const bool first_player = 0,
	const uint_fast8_t depth = np_sdepth, const bool chain_games = 0, const bool chain_player = 0, const bool print = 0);

uint_fast8_t np_minimax(const board& board, const bool max, const bool player, const uint_fast8_t move, const uint_fast8_t depth,
	const _c4_brain::c4_brain* brain,
	uint_fast8_t alpha = std::numeric_limits<uint8_t>::min(), uint_fast8_t beta = std::numeric_limits<uint8_t>::max());

int_fast8_t np_ai_play_ai(const _c4_brain::c4_brain* brain1, const _c4_brain::c4_brain* brain2,
	uint_fast8_t depth1, uint_fast8_t depth2, const bool chain_games = 0, const bool chain_player = 0, bool print = 0);

int_fast8_t np_i_play_ai(const _c4_brain::c4_brain* brain, uint_fast8_t depth, bool ai_first);

namespace _non_parallel_helper_foo {
	void clear_seen_boards();
}