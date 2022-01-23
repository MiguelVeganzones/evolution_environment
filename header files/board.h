#pragma once
#include <iostream>
#include <bitset>
#include <array>
#include <vector>

namespace b_const {
	static constexpr uint_fast8_t sx = 7, sy = 6, sn = 4;
	static constexpr uint_fast8_t size = sx * sy;
	static constexpr uint_fast8_t players = 2;
	extern const uint64_t mask;
}

using namespace b_const;

struct board
{
	board() {}
	board(const std::array<uint_fast8_t, size>& arr) 
	{
		for (uint_fast8_t i = 0; i < size; ++i)
		{
			state[0][i] = arr[i] == 1;//must be changed for more than two players
			state[1][i] = arr[i] == 2;//must be changed for more than two players
		}
	}

	board(const board& board, const uint_fast8_t move, const uint_fast8_t player) :state{ board.state }
	{
		state[player][move] = true;
	}

	//returns 1 if player 0, -1 if player 1 and 0 if no player has moved in position j, i
	inline int8_t operator()(const uint_fast8_t j, const uint_fast8_t i) const 
		{ return (state[0][j * sx + i] - state[1][j * sx + i]); } 

	std::array<std::bitset<size>, 2> state;

	const std::pair<uint64_t, uint64_t> to_num() const;
	const bool check_move (const uint_fast8_t move, const uint_fast8_t player) const; //player 0 or 1, index for state
	const std::vector<uint_fast8_t> get_moves() const;
	void make_move(const uint_fast8_t move, const uint_fast8_t player);
	friend std::ostream& operator <<(std::ostream& out, const board& board);

};

const std::array<std::bitset<size>, players> from_num(const std::pair<uint64_t, uint64_t>& nums); 