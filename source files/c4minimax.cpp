#include <array>
#include <iostream>
#include <thread>
#include <vector>
#include <shared_mutex>
#include "board.h"
#include "c4minimax.h"
#include "Random.h"
#include "stopwatch.h"
#include <algorithm>

//static std::map<std::pair<uint64_t, uint64_t>, uint_fast8_t> computed_boards;

static uint64_t count;

std::shared_mutex lock;

static constexpr const std::array<bool, 2> next_player{ 1, 0 };

void print_vector_pair(const std::vector<std::pair<uint_fast8_t, uint_fast8_t>>& v)
{
	for (auto& elem : v)
	{
		std::cout << " weight: " << (int)elem.first << ", move: " << (int)elem.second << "\n";
	}
}

uint_fast8_t c4_minimax(const board& prev_board, const bool first_player, const bool out)
{
	const auto moves = prev_board.get_moves();
	std::vector<std::unique_ptr<std::thread>> threads{};
	std::vector<std::pair<uint_fast8_t, uint_fast8_t>> main_w_and_m(moves.size());

	if (moves.size() == 0) { return 255; }

	uint_fast8_t i = 0;
	for (auto move : moves) {
		threads.push_back(std::make_unique<std::thread>(std::thread
			(minimax_interface, prev_board, first_player, move, sdepth-1, i++, std::ref(main_w_and_m))));
		//minimax_interface( prev_board, first_player, move, 1, i++, std::ref(main_w_and_m));
	}

	for (auto& t : threads) {
		t->join();
	}

	if (out) { std::cout << " minimax calls aprox: " << count << "\n" << " Unique boards stored: " << computed_boards.size() << "\n"; }

	print_vector_pair(main_w_and_m);

	computed_boards.clear();
	count = 0;
	//print(main_w_and_m);

	return std::max_element(main_w_and_m.begin(), main_w_and_m.end())->second;

}

void minimax_interface(const board& board, const bool player, const uint_fast8_t move, const uint_fast8_t depth, const uint_fast8_t i,
	std::vector<std::pair<uint_fast8_t, uint_fast8_t>>& main_w_and_m)
{
	main_w_and_m[i] = { minimax(board, 0, player, move, depth) , move };
}

uint_fast8_t insert_and_return(const std::pair<uint64_t,uint64_t> num_cur_board, const uint_fast8_t weight)
{
	std::unique_lock<std::shared_mutex> write_lock(lock);
	computed_boards.insert({ num_cur_board, weight });
	//std::cout << (int)weight << std::endl;
	return weight;
}

uint_fast8_t minimax(const board& prev_board, const bool max, const bool player, const uint_fast8_t move, const uint_fast8_t depth, uint_fast8_t alpha, uint_fast8_t beta)
  {
	++count;
	const board current_board(prev_board, move, player);
	const auto num_cur_board = current_board.to_num();
	//std::cout << current_board << std::endl;

	{
		std::shared_lock < std::shared_mutex > read_lock(lock);

		const auto it = computed_boards.find(num_cur_board);

		if (it != computed_boards.end()){
			//std::cout << "dict: " << (int)it->second << std::endl;
			return  it->second;
		}
	}

	if (current_board.check_move(move, player)) {
		return insert_and_return(num_cur_board, !max * 255); //!max because this move was done by previous  player
	}

	else if (depth == 0) {
		return insert_and_return(num_cur_board, (num_cur_board.first + num_cur_board.second) % 254 + 1); //Random::randint(1, 254));
	}

	else
	{
		const auto next_moves = current_board.get_moves();
		const uint_fast8_t branches = (uint_fast8_t)next_moves.size();

		if (branches == 0) { return insert_and_return(num_cur_board, 127); }

		bool flag = false; //if a branch of a node was cut, the weight of that node is not the real weight and thus should not be stored
		uint_fast8_t i = branches;

		if (max) {
			alpha = std::numeric_limits<uint8_t>::min();
			for (auto branch_move : next_moves) {
				alpha = std::max(alpha, minimax(current_board, !max, next_player[player], branch_move, depth - 1, alpha, beta));
				--i;
				if (beta <= alpha) {
					//std::cout << "Cut!\n";
					if (i != 0) { flag = true; } //unless no branches where cut, do not store the result
					break;
				}
			}
			return flag ? alpha : insert_and_return(num_cur_board, alpha);
		}

		else {
			beta = std::numeric_limits<uint8_t>::max();
			for (auto branch_move : next_moves) {
				beta = std::min(beta, minimax(current_board, !max, next_player[player], branch_move, depth - 1, alpha, beta));
				--i;
				if (beta <= alpha) {
					//std::cout << "Cut!\n";
					if (i != 0) { flag = true; } //unless no branches where cut, do not store the result
					break;
				}
			}
			return flag ? beta : insert_and_return(num_cur_board, beta);
		}
	}
}


uint_fast8_t ai_play_ai(const bool out) //returns 0 if draw, 1 if player one won, 2 if player 2 won
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
	//0,0,0,0,
	//0,0,0,0,
	//0,0,0,0,
	//0,0,0,0
	//};

//	std::array<uint_fast8_t, 9> arr{
//0,0,0,
//0,0,0,
//0,0,0
//	};

	board game_state(arr);

	//Random::init();

	std::array<bool, 2> next_player{ 1,0 };
	bool player = 0;

	while (1)
	{
		std::cout << "===================================================================\n\n";
		if (out) { stopwatch s; }

		uint_fast8_t move = c4_minimax(game_state, player, out);

		if (move == 255) { if (out) { std::cout << "It was a draw\n"; } return 0; } // return 0 because it was a draw

		game_state.make_move(move, player);
		if(out) std::cout << game_state << std::endl;

		if (game_state.check_move(move, player)) { if (out) { std::cout << "player " << player + 1 << " wins.\n"; } return player + 1; }

		player = next_player[player];
		std::cout << "===================================================================\n\n";
	}
}

void i_play_ai(const bool ai_first)
{
	stopwatch global;

	std::array<uint_fast8_t, 42> arr{
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,
	};

	//std::array<uint_fast8_t, 16> arr{
	//0,0,0,0,
	//0,0,0,0,
	//0,0,0,0,
	//0,0,0,0
	//};

//	std::array<uint_fast8_t, 9> arr{
//0,0,0,
//0,0,0,
//0,0,0
//	};

	board game_state(arr);

	//Random::init();

	std::array<bool, 2> next_player{ 1,0 };
	bool player = ai_first;

	while (1)
	{
		uint_fast8_t move;

		if(player)
		{
			stopwatch s;
			move = c4_minimax(game_state, ai_first, 1);
		}

		else	 
		{
			const auto moves = game_state.get_moves();
			if (moves.size() == 0) { std::cout << "No more moves\n"; break; }
			std::cout << "Choose a move: ";

			for (int i = moves.size() - 1; i > -1; --i)
			{
				std::cout << (unsigned int)moves[i] << ", ";
			}
			
			std::cout<<std::endl;

			std::cout << "Move: ";
			std::cin >> move;

			while (std::find(moves.begin(), moves.end(), move) == moves.end())
			{
				std::cout << "Invalid move: "<<move<<"\n";
				std::cout << "Try another move: ";
				std::cin >> move;
			}
		}

		game_state.make_move((uint_fast8_t)move, player);
		std::cout << game_state << std::endl;
		if (game_state.check_move(move, player)) { std::cout << "player " << player + 1 << " wins.\n"; break; }

		player = next_player[player];

	}

}


//bool operator<(const computed_board& first, const computed_board& second)
//{
//	auto first_first = first.first.first;
//	auto second_first = second.first.first;
//	if (first_first != second_first)
//		return first_first < second_first;
//
//	return first.first.second < second.first.second;
//}


//single thread ~

//uint_fast8_t c4_minimax(const board& prev_board, const bool first_player)
//{
//	const auto moves = prev_board.get_moves();
//	std::vector<std::pair<uint_fast8_t, uint_fast8_t>> main_w_and_m(moves.size());
//	Random::init();
//
//	uint_fast8_t i = 0;
//	for (auto move : moves)
//	{
//		uint_fast8_t weight = minimax(prev_board, first_player, move, 0);
//		//if (weight == first_player * 255) { return move; }
//		main_w_and_m[i++] = { weight , move };
//	}
//
//	std::cout << " minimax calls aprox: " << count << "\n";
//
//	std::sort(main_w_and_m.begin(), main_w_and_m.end());
//	if (first_player) { return(main_w_and_m.back().second); }
//	else { return(main_w_and_m.front().second); }
//}


	//else
	//{
	//const auto next_moves = current_board.get_moves();

	//if (next_moves.size() == 0) { return insert_and_return(num_cur_board, 127); }

	//std::vector<uint_fast8_t> weights(next_moves.size());

	//uint_fast8_t i = 0;
	//for (auto branch_move : next_moves) {
	//	weights[i++] = minimax(current_board, !max, next_player[player], branch_move, depth + 1);
	//}

	//if (max) { return insert_and_return(num_cur_board, *std::max_element(weights.begin(), weights.end())); }
	//else { return insert_and_return(num_cur_board, *std::min_element(weights.begin(), weights.end())); }
	//}