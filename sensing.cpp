#include "types.h"

using namespace argos;

class Sensing{


public:
	char predict(const CCI_PositioningSensor::SReading& robot_state){
		char zone;
		for (const auto& [key, bboxes] : ground_truth){
			for (auto bbox : bboxes){
				if (robot_state.Position.GetX() > bbox[0] && robot_state.Position.GetX() < bbox[4]
					&& robot_state.Position.GetY() > bbox[5] && robot_state.Position.GetY() < bbox[1]){
					zone = key;
					break;
				}
			}
		}
		
		return zone;
	}

	std::tuple<CVector2, bool> getGoal(const CCI_PositioningSensor::SReading& robot_state, char zone_old, char zone){
		CVector2 goal_state;
		bool manouvering = false;

		int index = -1;
		Real d = 300.0;
	   
		if (zone_old == 'C' && zone == 'I'){
			std::array<CVector2,3> goal_pool;
			for (int i=0;i<8;i=i+2){
				if ((CVector2(robot_state.Position.GetX(), robot_state.Position.GetY()) - CVector2(ground_truth[zone][0][i],ground_truth[zone][0][i+1])).Length() < d){
					d = (CVector2(robot_state.Position.GetX(), robot_state.Position.GetY()) - CVector2(ground_truth[zone][0][i],ground_truth[zone][0][i+1])).Length();
					index = i;
				}
			}

			goal_pool[0] = CVector2(ground_truth[zone][0][(index%8)], ground_truth[zone][0][(index+1)%8]);
			goal_pool[1] = CVector2(ground_truth[zone][0][(index+2)%8], ground_truth[zone][0][(index+3)%8]);
			goal_pool[2] = CVector2(ground_truth[zone][0][(index+4)%8], ground_truth[zone][0][(index+5)%8]);
	      
			std::random_device rd;
			std::default_random_engine eng(rd());
			std::uniform_int_distribution<int> distr(0, 2);

			goal_state = goal_pool[distr(eng)];
			manouvering = true;


			//std::cout << this->goal_state << "\n";
		}

		if (zone_old == 'C' && zone == 'G'){
			std::array<CVector2,2> goal_pool;
			for (int i=0;i<8;i=i+2){
				if ((CVector2(robot_state.Position.GetX(), robot_state.Position.GetY()) - CVector2(ground_truth[zone][0][i],ground_truth[zone][0][i+1])).Length() < d){
					d = (CVector2(robot_state.Position.GetX(), robot_state.Position.GetY()) - CVector2(ground_truth[zone][0][i],ground_truth[zone][0][i+1])).Length();
					index = i;
				}
			}

			if (index == 2){
				goal_pool[0] = CVector2(ground_truth[zone][0][index], ground_truth[zone][0][index+1]);
				goal_pool[1] = CVector2(ground_truth[zone][0][4], ground_truth[zone][0][5]);
			}
			else if(index == 4){
				goal_pool[0] = CVector2(ground_truth[zone][0][index], ground_truth[zone][0][index+1]);
				goal_pool[1] = CVector2(ground_truth[zone][0][0], ground_truth[zone][0][1]);
			}
			else if(index == 6){
				goal_pool[0] = CVector2(ground_truth[zone][0][0], ground_truth[zone][0][1]);
				goal_pool[1] = CVector2(ground_truth[zone][0][2], ground_truth[zone][0][3]);
			}

			std::random_device rd;
			std::default_random_engine eng(rd());
			std::uniform_int_distribution<int> distr(0, 1);

			goal_state = goal_pool[distr(eng)];
			manouvering = true;

			//std::cout << this->goal_state << "\n";
		}

		return std::make_tuple(goal_state, manouvering);
	}



};