//#include <iostream>
#include <string>
#include <algorithm>
#include <bitset>
#include <vector>
#include "board.h"
#include <assert.h>

static const uint64_t b_const::mask = (const uint64_t)pow(2, sx * sy) - 1;
using namespace b_const;

const bool board::check_move(const  uint_fast8_t move, const bool player) const
{
	const uint_fast8_t n = sn - 1; //fichas adyacentes necesarias
	const uint_fast8_t x = move % sx;
	const uint_fast8_t y = move / sx;
	auto _state = state[player];
	_state[move] = 1;

	short int i;
	uint_fast8_t _x, _y;

	//comprobacion diagonal \
	// i = num pasos en diagonal
	uint_fast8_t count = 0;
	for (i = -std::min({ x, y, n }); i < std::min({ (uint_fast8_t)(sx - x), (uint_fast8_t)(sy - y), sn }); ++i)
	{
		_y = y + i;
		_x = x + i;
		//std::cout << (y * sx + x + (sx + 1) * i) << std::endl;
		if (_state[_y * sx + _x]) {
			++count; 
			if (count == sn) { return true; }
		}

		else { count = 0; }
	}

	//comprobacion diagonal /
	// i = num pasos en diagonal
	count = 0;
	for (i = -std::min({ y,(uint_fast8_t)(sx - x - 1), n }); i < std::min({ (uint_fast8_t)(sy - y), (uint_fast8_t)(x + 1), sn }); ++i)
	{
		_y = y + i;
		_x = x - i;
		//std::cout << (y * sx + x + (sx - 1) * i) << std::endl;
		if(_state[_y * sx + _x]){
			++count;
			if (count == sn) { return true; }
		}

		else { count = 0; }
	}

	//comprobacion horizontal --
	count = 0;
	for (i = std::max({ x - n, 0 }); i < std::min({ (uint_fast8_t)(x + sn), sx }); ++i)
	{
		//std::cout << (unsigned int)(y * sx + i) << std::endl;
		if (_state[y * sx + i]) {
			++count;
			if (count == sn) { return true; }
		}

		else { count = 0; }
	}

	//comprobacion vertical |
	if (y < sy - n)
	{
		count = 0;
		for (i = 1; i < sn; ++i)
		{
			//std::cout << (unsigned int)(i * sx + move) << std::endl;
			if (_state[i * sx + move] == false) { return false; }
		}
		return true;
	}

	return false;
}

const std::vector<uint_fast8_t> board::get_moves() const
{
	std::vector<uint_fast8_t> possible_moves{}; // si no hay posicion, coloca un valor negativo

	uint_fast8_t position;
	for (int i = sx - 1; i > -1; --i)
	{
		for (int j = sy - 1; j > -1; --j)
		{
			position = j * sx + i;
			//std::cout << (int)i << " " << int(ii) << " " << (int)state[0][position] << " " << (int)state[1][position] << std::endl;
			if (!(state[0][position] | state[1][position]))
			{
				possible_moves.push_back(position);
				break;
			}
		}
	}

	return possible_moves;
}

const std::array<std::bitset<size>, players> from_num(const std::pair<uint64_t, uint64_t>& nums)
{
	return { *(std::bitset<size>*) & nums.first, *(std::bitset<size>*) & nums.second };
}

const std::pair<uint64_t, uint64_t> board::to_num() const
{
	//even tho the states are std::bitset<42>, they are stored as 8 bytes
	return { (*(uint64_t*)&state[0] & mask) , (*(uint64_t*)&state[1] & mask) }; //evil bit hacking
}

std::ostream& operator <<(std::ostream& out, const board& board)
{
	for (int i = 0; i < sy; ++i) {
		for (int ii = 0; ii < sx; ++ii) {
			const uint_fast8_t position = i * sx + ii;
			if (board.state[0][position]) {
				out << "o,";
			}
			else if (board.state[1][position]) {
				out << "x,";
			}
			else
				out << " ,";
		}
		out << "\n";
	}
	out << std::flush;
	return out;
}

void board::make_move(const uint_fast8_t move, const uint_fast8_t player)
{
	//auto temp = state[player];
	//temp[move] = 1;
	//state[player] = temp;

	assert(!(state[0][move] | state[1][move]));
	state[player][move] = 1;
}


//const bool board::check_move(const  uint_fast8_t move, const bool player) const
//{
//	//Para el yonki del futuro:
//	//esta funcion no funciona bien, tienes que crear una lista nueva y mirar esa.
//	// no es buena idea penandolo, pero no funciona no
//
//
//	const uint_fast8_t n = sn - 1; //fichas adyacentes necesarias
//	const uint_fast8_t x = move % sx;
//	const uint_fast8_t y = move / sx;
//	auto _state = state[player];
//	_state[move] = 1;
//
//	//comprobacion diagonal \
//	// i = num pasos en diagonal
//	uint_fast8_t count = 0;
//	for (int i = -std::min({ x, y, n }); i < std::min({ (uint_fast8_t)(sx - x), (uint_fast8_t)(sy - y), sn }); ++i)
//	{
//		//std::cout << (y * sx + x + (sx + 1) * i) << std::endl;
//		if (_state[y * sx + x + (sx + 1) * i]) {
//			++count;
//			if (count == sn) { return true; }
//		}
//
//		else { count = 0; }
//	}
//
//	//comprobacion diagonal /
//	// i = num pasos en diagonal
//	count = 0;
//	for (int i = -std::min({ y,(uint_fast8_t)(sx - x - 1), n }); i < std::min({ (uint_fast8_t)(sy - y), (uint_fast8_t)(x + 1), sn }); ++i)
//	{
//		//std::cout << (y * sx + x + (sx - 1) * i) << std::endl;
//		if (_state[y * sx + x + (sx - 1) * i]) {
//			++count;
//			if (count == sn) { return true; }
//		}
//
//		else { count = 0; }
//	}
//
//	//comprobacion horizontal --
//	count = 0;
//	for (uint_fast8_t i = std::max({ x - n, 0 }); i < std::min({ (uint_fast8_t)(x + sn), sx }); ++i)
//	{
//		//std::cout << (unsigned int)(y * sx + i) << std::endl;
//		if (_state[y * sx + i]) {
//			++count;
//			if (count == sn) { return true; }
//		}
//
//		else { count = 0; }
//	}
//
//	//comprobacion vertical |
//	if (y < sy - n)
//	{
//		count = 0;
//		for (uint_fast8_t i = 1; i < sn; ++i)
//		{
//			//std::cout << (unsigned int)(i * sx + move) << std::endl;
//			if (_state[i * sx + move] == false) { return false; }
//		}
//		return true;
//	}
//
//	return false;
//}

