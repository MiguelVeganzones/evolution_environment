#pragma once
#include <thread>
#include <iostream>
#include <map>
#include "board.h"
#include <vector>
#include "parallel_hashmap/phmap.h"

static constexpr uint_fast8_t sdepth = 10; //calls to minimax  <= sx + sx**2 + .... + sx**sdepth (creo)
//phmap::parallel_flat_hash_map

static phmap::parallel_flat_hash_map<std::pair<uint64_t, uint64_t>, uint_fast8_t> computed_boards;

uint_fast8_t c4_minimax(const board& prev_board, const bool first_player = 0, const bool out = 1);

uint_fast8_t minimax(const board&, const bool max, const bool player, const uint_fast8_t move, const uint_fast8_t depth,
	uint_fast8_t alpha = std::numeric_limits<uint_fast8_t>::min(), uint_fast8_t beta = std::numeric_limits<uint_fast8_t>::max());

void minimax_interface(const board& board, bool player, const uint_fast8_t move, const uint_fast8_t depth, const uint_fast8_t i,
	std::vector<std::pair<uint_fast8_t, uint_fast8_t>>& main_m_and_w);

uint_fast8_t ai_play_ai(const bool out=1);

void i_play_ai(const bool ai_first = false); //ai_first = 0 for i, 1 for ai

//bool operator<(const computed_board& first, const computed_board& second);