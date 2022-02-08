#include "types.h"

using namespace argos;


Real sgn(Real v){
    if (v >= 0.0) {return 1.0;}
    return 0.0;
}

std::array<Real, 2> WallFollowing(std::map<CRadians, struct angle_data> world_model_long){

    Real r_distance_d = 35.0;
    CRadians r_orientation_d = -CRadians::PI_OVER_TWO;

   	//Wall Following-----------------------------------------
	Real v_wf, w_wf, modulation;
	Real distance_error, orientation_error;
	CRange F_range(-CRadians::PI_OVER_TWO, CRadians::PI_OVER_TWO);
	CRange R_range(-CRadians::PI, CRadians::ZERO);

	angle_data R_min = {CRadians::ZERO, 150.0, 0, false};
	angle_data F_min = {CRadians::ZERO, 150.0, 0, false};

	for (const auto& [angle, data] : world_model_long){
		if (F_range.WithinMinBoundIncludedMaxBoundIncluded(angle)){
		    if (data.occluded == true){
                if (data.distance <= F_min.distance){
                    F_min.distance = data.distance;
                    F_min.angle = data.angle;
                }
            }
		}

		if (R_range.WithinMinBoundIncludedMaxBoundIncluded(angle)){
			if (data.occluded == false){
				if (data.distance <= R_min.distance){
					R_min.distance = data.distance;
					R_min.angle = data.angle;
				}
			}
		}
	}

	//std::cout << F_min.angle << " " << F_min.distance << "\n";
	//fprintf(stderr, "Angle: %f - Distance: %f\n", F_min.angle.GetValue(), F_min.distance);


   	// linear velocity wf base + linearly ramping down to 0 component (Arrival)
   	if (F_min.distance >= 15.0 && F_min.distance <= 60.0)
   	    modulation = ((F_min.distance - 15.0)/(60.0 - 15.0));
   	else if (F_min.distance < 15.0)
   	    modulation = 0.0;
   	else if (F_min.distance > 60.0)
   	    modulation = 1.0;

   	v_wf = 2.0 + 6.0 * modulation;
   	// angular velocity wf proportional to distance error + orientation error
   	w_wf = 0.01*(r_distance_d - R_min.distance) + (-(r_orientation_d - R_min.angle).SignedNormalize().GetValue());


   	return {v_wf, w_wf};
}

std::array<Real, 2> ObstacleAvoidance(const CCI_RangeAndBearingSensor::TReadings& rab_readings){
   	//Obstacle Avoidance--------------------------------------
	Real center_center_distance = 17.0; //cm distanza tra i centri di due footbot attaccati
	Real alpha_max = 30.0; //cm
	Real w_oa;
	CVector2 ahead = CVector2(1.0,0.0) * alpha_max;
	CVector2 rab_xy, nearest_robot_xy;
	std::vector<CCI_RangeAndBearingSensor::SPacket> in_rectangle_robots;

	for(auto rr : rab_readings){
      	rab_xy.FromPolarCoordinates(rr.Range - center_center_distance, rr.HorizontalBearing);
      	if(rab_xy.GetX() > 0.0 and rab_xy.GetX() <= ahead.GetX() and abs(rab_xy.GetY()) <= center_center_distance/2){
         	in_rectangle_robots.push_back(rr);
      	}
      
   	}

	Real nearest_robot_range = CVector2(ahead.GetX(), center_center_distance/2).Length();
	CRadians nearest_robot_angle;
	for(auto irr : in_rectangle_robots){
		if(irr.Range <= nearest_robot_range){
			nearest_robot_range = irr.Range;
			nearest_robot_angle = irr.HorizontalBearing;
		}
   	}

   	nearest_robot_xy.FromPolarCoordinates(nearest_robot_range - center_center_distance, nearest_robot_angle);
   	//this->nearest_robot_xy = nearest_robot_xy;

   	// angular velocity oa
   	w_oa = (ahead + CVector2(0.0, -nearest_robot_xy.GetY())).Angle().GetValue();

   	return {0.0, w_oa};
}

std::array<Real, 2> Crossing(const CCI_PositioningSensor::SReading& robot_state, std::map<CRadians, struct angle_data> world_model_long, CVector2 goal_state){
	Real v_c, w_c, modulation;
	CRadians X,Y,theta;
	CVector2 robot_position(robot_state.Position.GetX(), robot_state.Position.GetY());
	robot_state.Orientation.ToEulerAngles(theta,Y,X);
	CRange F_range(-CRadians::PI_OVER_SIX, CRadians::PI_OVER_SIX);
	
	angle_data F_min = {CRadians::ZERO, 150.0, 0, false};
	for (const auto& [angle, data] : world_model_long){
		if (F_range.WithinMinBoundIncludedMaxBoundIncluded(angle)){
		    if (data.occluded == true){
                if (data.distance <= F_min.distance){
                    F_min.distance = data.distance;
                    F_min.angle = data.angle;
                }
            }
		}
	}
	


   	// linear velocity wf base + linearly ramping down to 0 component (Arrival)
   	if (F_min.distance >= 15.0 && F_min.distance <= 60.0)
   	    modulation = ((F_min.distance - 15.0)/(60.0 - 15.0));
   	else if (F_min.distance < 15.0)
   	    modulation = 0.0;
   	else if (F_min.distance > 60.0)
   	    modulation = 1.0;

   	v_c = 2.0 + 6.0 * modulation;
	w_c = 0.1*((goal_state - robot_position).Angle() - theta).SignedNormalize().GetValue();

	return {v_c, w_c};
}







	// std::array<Real,2> UnstructuredExploration(
	//    const CCI_FootBotProximitySensor::TReadings& proximity_readings){

	//    /* Sum them together */
	//    Real v_l, v_r;
	//    CVector2 cAccumulator;

	//    for(int i = 0; i < proximity_readings.size(); ++i) 
	//       cAccumulator += CVector2(proximity_readings[i].Value, proximity_readings[i].Angle);
	   
	//    cAccumulator /= proximity_readings.size();
	   

	//    if (cAccumulator.Length() == 0.0){
	//       if(tic % 30 == 0 || chosen){
	         
	//          if (!chosen){
	//             std::random_device rd;
	//             std::default_random_engine eng(rd());
	//             std::uniform_real_distribution<float> distr(0, 1);
	            
	//             if (distr(eng) >= 0.5)
	//                choice = 0;
	//             else
	//                choice = 1;

	//             chosen = true;
	//          }

	//          if (counter < 10){
	//             if (choice == 0){
	//                v_l = 2.0;
	//                v_r = -2.0;
	//             }
	//             else{
	//                v_l = -2.0;
	//                v_r = 2.0;
	//             }
	//          counter++;
	//          }
	         
	//          else{
	//             counter = 0;
	//             chosen = false;
	//          }
	         
	//       }
	//       else{
	//          v_l = 5.0;
	//          v_r = 5.0;
	//       }
	//    }
	//    else{
	//       if(m_cGoStraightAngleRange.WithinMinBoundIncludedMaxBoundIncluded(cAccumulator.Angle())) {
	//          if(cAccumulator.Angle().GetValue() > 0.0f) {
	//             v_l = 2.0;
	//             v_r = 0.0;
	//          }
	//          else {
	//             v_l = 0.0;
	//             v_r = 2.0;
	//          }
	//       }
	//       else{
	//          v_l = 5.0;
	//          v_r = 5.0;
	//       }
	//    }


	//    return {v_l, v_r};
	// }
