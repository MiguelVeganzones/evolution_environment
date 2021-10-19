#include <array>
#include <iostream>
#include <vector>
#include "board.h"
#include "non_parallel_minimax.h"
#include "Random.h"
#include "stopwatch.h"
#include "c4_brain.h"

static const uint_fast8_t np_sdepth = 10; //calls to minimax  <= sx + sx**2 + .... + sx**sdepth (creo)
static thread_local phmap::parallel_flat_hash_map<std::pair<uint64_t, uint64_t>, uint_fast8_t> np_computed_boards;
static thread_local phmap::parallel_flat_hash_map<std::pair<uint64_t, uint64_t>, uint_fast8_t> np_seen_boards;
//static unsigned int counter = 0;

static uint64_t np_count = 0;

static constexpr const std::array<bool, 2> next_player{ 1, 0 };

inline void np_print_vector_pair(const std::vector<std::pair<uint_fast8_t, uint_fast8_t>>& v)
{
	for (auto& elem : v)
	{
		std::cout << " weight: " << (int)elem.first << ", move: " << (int)elem.second << "\n";
	}
	std::cout << "\n";
}

uint_fast8_t np_c4_minimax(const board& prev_board, const _c4_brain::c4_brain* brain, const bool first_player,
	const uint_fast8_t depth, const bool chain_games, const bool chain_player, const bool print)
{
	if ((first_player == chain_player) and chain_games) {
		const auto it = np_seen_boards.find(prev_board.to_num());

		if (it != np_seen_boards.end()) {
			//std::cout << prev_board << std::endl;
			//++counter;
			return  it->second;
		}
	}
	
	const auto moves = prev_board.get_moves();

	if (moves.size() == 0) { return 255; } //move 255 used for a draw
	std::vector<std::pair<uint_fast8_t, uint_fast8_t>> main_w_and_m(moves.size());

	uint_fast8_t i = 0;
	for (auto move : moves)
	{
		const auto weight = np_minimax(prev_board, 0, first_player, move, depth - 1, brain);
		//if (weight == 255) { return move; }
		main_w_and_m[i++] = { weight, move };
	}

	//std::cout << " minimax calls aprox: " << np_count << "\n" << " Unique boards stored: " << np_computed_boards.size() << "\n";

	np_computed_boards.clear();
	np_count = 0;

	if (print) np_print_vector_pair(main_w_and_m);

	const auto ret = std::max_element(main_w_and_m.begin(), main_w_and_m.end())->second;

	if ((first_player == chain_player) and chain_games) 
		np_seen_boards.insert({ prev_board.to_num(), ret });

	return ret;
}

inline uint_fast8_t np_insert_and_return(const std::pair<uint64_t, uint64_t> num_cur_board,
	const uint_fast8_t weight)
{
	np_computed_boards.insert({ num_cur_board, weight });
	//std::cout << (int)weight << std::endl;
	return weight;
}

uint_fast8_t np_minimax(const board& prev_board, const bool max, const bool player, const uint_fast8_t move,
	const uint_fast8_t depth, const _c4_brain::c4_brain* brain, uint_fast8_t alpha, uint_fast8_t beta)
{
	++np_count;
	const board current_board(prev_board, move, player);
	const auto num_cur_board = current_board.to_num();

	//std::cout << "a: " << (int)alpha << "  ,b: " << (int)beta << std::endl;
	//std::cout << current_board << std::endl;

	const auto it = np_computed_boards.find(num_cur_board);

	if (it != np_computed_boards.end()) {
		//std::cout << "dict: " << (int)it->second << std::endl;
		return  it->second;
	}

	else if (current_board.check_move(move, player)) {
		return np_insert_and_return(num_cur_board, !max * 255);//!max because this move was done by previous player
	}

	else if (depth == 0) {
		auto w = brain->weigh(current_board, player);
		//std::cout << (int)w << "\n";
		return np_insert_and_return(num_cur_board, w);
	}

	else
	{
		const auto next_moves = current_board.get_moves();
		const uint_fast8_t branches = (uint_fast8_t)next_moves.size();

		if (branches == 0) { return np_insert_and_return(num_cur_board, 127); }
		
		bool flag = false; //if a branch of a node was cut, the weight of that node is not the real weight and thus should not be stored
		uint_fast8_t i = branches;

		if (max) {
			alpha = std::numeric_limits<uint8_t>::min();
			for (auto branch_move : next_moves) {
				alpha = std::max(alpha, np_minimax(current_board, !max, next_player[player], branch_move, depth - 1, brain, alpha, beta));
				--i;
				if (beta <= alpha) {
					//std::cout << "Cut!\n";
					if (i != 0) { flag = true; } //unless no branches where cut, do not store the result
					break;
				}
			}
			return flag ? alpha : np_insert_and_return(num_cur_board, alpha);
		}
	
		else {
			beta = std::numeric_limits<uint8_t>::max();
			for (auto branch_move : next_moves) {
				beta = std::min(beta, np_minimax(current_board, !max, next_player[player], branch_move, depth - 1, brain, alpha, beta));
				--i;
				if (beta <= alpha) {
					//std::cout << "Cut!\n";
					if (i != 0) { flag = true; }//unless no branches where cut, do not store the result
					break;
				}
			}
			return flag ? beta : np_insert_and_return(num_cur_board, beta);
		}
	}
}

short int np_ai_play_ai(const _c4_brain::c4_brain* brain1, const _c4_brain::c4_brain* brain2,
	uint_fast8_t depth1, uint_fast8_t depth2, const bool chain_games, const bool chain_player, bool print)
{
	if (print) {
		stopwatch global;
		std::ios_base::sync_with_stdio(false);
	}

	std::array<uint_fast8_t, 42> arr{
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
	};

	//std::array<uint_fast8_t, 16> arr{
	//	0,0,0,0,
	//	0,0,0,0,
	//	0,0,0,0,
	//	0,0,0,0
	//	};

	//std::array<uint_fast8_t, 9> arr{
	//0,0,0,
	//0,0,0,
	//0,0,0
	//}

	board _board(arr);

	//Random::init();

	std::array<bool, 2> next_player{ 1,0 };
	bool player = 0;

	const _c4_brain::c4_brain* p_brain;
	uint_fast8_t depth;

	while (1)
	{
		p_brain = player ? brain2 : brain1;
		depth = player ? depth2 : depth1;

		if(print) std::cout << "===================================================================\n\n";
		//stopwatch s;

		const uint_fast8_t move = np_c4_minimax(_board, p_brain, player, depth, chain_games, chain_player, print);

		if (move == 255) { 
			if(print) std::cout << "It was a draw\n"; 
			return -1;  // return -1 because it was a draw
		}

		_board.make_move(move, player);
		if(print) std::cout << _board << std::endl;

		if (_board.check_move(move, player)) {
			if(print) std::cout << "player " << player + 1 << " wins.\n"; 
			break; 
		}

		player = next_player[player];
		if(print) std::cout << "===================================================================\n\n";
	}

	if(print) std::cout << "done\nDepth 1: " << (int)depth1 << "\nDepth 2: " << (int)depth2 << std::endl;
	return player;
}

short int np_i_play_ai(const _c4_brain::c4_brain* brain, uint_fast8_t depth, bool ai_first)
{
	stopwatch global;

	std::ios_base::sync_with_stdio(false);

	std::array<uint_fast8_t, 42> arr{
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
	};

	//std::array<uint_fast8_t, 16> arr{
	//	0,0,0,0,
	//	0,0,0,0,
	//	0,0,0,0,
	//	0,0,0,0
	//	};

	//std::array<uint_fast8_t, 9> arr{
	//0,0,0,
	//0,0,0,
	//0,0,0
	//}

	board _board(arr);

	//Random::init();

	std::array<bool, 2> next_player{ 1,0 };
	bool player = 0;

	while (1)
	{
		std::cout << "===================================================================\n\n";
		stopwatch s;

		unsigned short int move;
		
		if (ai_first) {
			stopwatch s;
			move = np_c4_minimax(_board, brain, player, depth);
		}

		else {
			const auto moves = _board.get_moves();
			if (moves.size() == 0) { std::cout << "No more moves\n"; break; }

			for (int i = moves.size() - 1; i > -1; --i) {
				std::cout << (unsigned int)moves[i] << ", ";
			}

			std::cout << std::endl;

			std::cout << "Move: ";
			std::cin >> move;

			while (std::find(moves.begin(), moves.end(), move) == moves.end())
			{
				std::cout << "Invalid move: " << move << "\n";
				std::cout << "Try another move: ";
				std::cin >> move;
			}

		}

		ai_first = !ai_first;

		if (move == 255) {
			std::cout << "It was a draw\n";
			return -1;  // return -1 because it was a draw
		}

		_board.make_move(move, player);
		std::cout << _board << std::endl;

		if (_board.check_move(move, player)) {
			std::cout << "player " << player + 1 << " wins.\n";
			break;
		}

		player = next_player[player];
		std::cout << "===================================================================\n\n";
	}

	std::cout << "done\n";
	return player;
}

uint_fast8_t heuristic(const board& b)
{
	const auto num_cur_board = b.to_num();
	return (num_cur_board.first + num_cur_board.second) % 254 + 1;
}

////	else
////	{
////	const auto next_moves = current_board.get_moves();
////
////	if (next_moves.size() == 0) { return np_insert_and_return(num_cur_board, 127); }
////
////	std::vector<uint_fast8_t> weights(next_moves.size());
////
////	if (max) {
////		alpha = std::numeric_limits<uint_fast8_t>::min();
////		for (auto branch_move : next_moves) {
////			alpha = std::max(alpha, np_minimax(current_board, !max, next_player[player], branch_move, depth + 1, alpha, beta));
////			if (beta <= alpha) {
////				//std::cout << "Cut!\n";
////				break;
////			}
////		}
////		return np_insert_and_return(num_cur_board, alpha);
////	}
////
////	else {
////		beta = std::numeric_limits<uint_fast8_t>::max();
////		for (auto branch_move : next_moves) {
////			beta = std::min(beta, np_minimax(current_board, !max, next_player[player], branch_move, depth + 1, alpha, beta));
////			if (beta <= alpha) {
////				//std::cout << "Cut!\n";
////				break;
////			}
////		}
////		return np_insert_and_return(num_cur_board, beta);
////	}
////	}
////}

void _non_parallel_helper_foo::clear_seen_boards()
{
	np_seen_boards.clear();
	//std::cout << counter << "\n";
	//counter = 0;
}
