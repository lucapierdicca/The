/* Include the controller definition */
#include "footbot_wall.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>

#include <algorithm>




CFootBotWall::CFootBotWall() : // initializer list
   m_pcWheels(NULL),
   m_pcDistanceS(NULL),
   m_pcDistanceA(NULL),
   m_pcProximity(NULL),
   m_pcRangeAndBearingS(NULL),
   m_pcPositioning(NULL),
   m_cAlpha(10.0f),
   m_fDelta(0.5f),
   m_fWheelVelocity(2.5f),   
   m_cGoStraightAngleRange(-ToRadians(m_cAlpha),
                           ToRadians(m_cAlpha)) {}

void CFootBotWall::Init(TConfigurationNode& t_node) {
   
   //Sensors & Actuators
   m_pcWheels     = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
   
   m_pcDistanceA  = GetActuator<CCI_FootBotDistanceScannerActuator>("footbot_distance_scanner");
   m_pcDistanceA->SetRPM(30.0);
   
   m_pcDistanceS  = GetSensor<CCI_FootBotDistanceScannerSensor>("footbot_distance_scanner");
   m_pcProximity  = GetSensor<CCI_FootBotProximitySensor>("footbot_proximity");
   m_pcRangeAndBearingS = GetSensor<CCI_RangeAndBearingSensor>("range_and_bearing");
   m_pcPositioning = GetSensor<CCI_PositioningSensor>("positioning");

   
   //Parse the configuration file
   GetNodeAttributeOrDefault(t_node, "alpha", m_cAlpha, m_cAlpha);
   m_cGoStraightAngleRange.Set(-ToRadians(m_cAlpha), ToRadians(m_cAlpha));
   GetNodeAttributeOrDefault(t_node, "delta", m_fDelta, m_fDelta);
   GetNodeAttributeOrDefault(t_node, "velocity", m_fWheelVelocity, m_fWheelVelocity);

   
   R.angle_interval = {-CRadians::PI, CRadians::ZERO};
   F.angle_interval = {-CRadians::PI_OVER_SIX/2.0, CRadians::PI_OVER_SIX/2.0};
   H.angle_interval = {-CRadians::PI, CRadians::PI};
   

   sectorLbl_to_sectorData = {{'R', R},
                              {'F', F},
                              {'H', H}};

   classLbl_to_template = {{'V',{0,2,0,1}},
                           {'C',{0,0,0,2}},
                           {'G',{0,1,2,0}},
                           {'I',{0,4,0,0}}};

}

Real CFootBotWall::EucDistance(std::array<int,4> u, std::array<int,4> v){
   Real euc_distance = 0.0;

   for(int i=0;i<u.size();i++)
      euc_distance = euc_distance + Square(u[i] - v[i]);
   
   return Sqrt(euc_distance);
}


bool isOpen(Real distance){
   if(distance >= 140.0)
      return true;
   else
      return false;
}


void CFootBotWall::getLocalMinMaxReadings(){
   
   // find local mins
   Real l ,c, r, ll, rr, lll, rrr; // sliding window |l|c|r|

   for(int i = 0;i<pr.size()-4;i++){
      ll = pr[i].distance;
      l = pr[i+1].distance;
      c = pr[i+2].distance;
      r = pr[i+3].distance;
      rr = pr[i+4].distance;

      if(ll>l && l>c && c<r && r<rr)
         lmr_new.push_back(pr[i+2]);
   }

   //find local maxs (dalle aperture)
   CRadians start, end; 
   int start_index, end_index, first_start_index, first_end_index;

   CRadians current_angle;
   Real current_distance;

   bool started = false;

   for(int i=0;i<pr.size();i++){
      current_angle = pr[i].angle;
      current_distance = pr[i].distance;

      if (started == false){ 
         if (isOpen(current_distance)){
            start = current_angle;
            start_index = i;
            end = start;
            end_index = start_index;
            started = true;
         }
      }

      if (started == true){
         if (isOpen(current_distance)){
            end = current_angle;
            end_index = i;
         }
         else{
            if (start_index == 0){
               first_start_index = start_index;
               first_end_index = end_index; //copy the first end index
            }

            lMr.push_back(pr[int(start_index + (end_index-start_index)/2.0)]);
            started = false;
         }
      }
   }

   if (started == true)
      if (first_start_index != 0)
         lMr.push_back(pr[int(start_index + (end_index-start_index)/2.0)]);
      else{
         lMr.erase(lMr.begin());
         lMr.push_back(pr[int(start_index + (end_index+first_end_index-start_index)/2.0) % pr.size()]);
      }
}



void CFootBotWall::processReadings(char sector_lbl){
   
   pr = sectorLbl_to_sectorData[sector_lbl].readings;

   Real avg;
   int window_len = 5;
   int readings_len = sectorLbl_to_sectorData[sector_lbl].readings.size();
   int j = 0;

   for(int i = 0;i<readings_len;i++){
      avg = 0.0f;
      j = 0;
      for(j;j<window_len;j++)
         avg = avg + sectorLbl_to_sectorData[sector_lbl].readings[(i+j) % readings_len].distance;
   
      avg = avg / window_len;

      pr[(i+int((window_len-1)/2)) % readings_len].distance = avg;
   }
}



std::pair<CRadians,Real> CFootBotWall::getMinReading(char sector_lbl){
   Real distance;
   std::pair <CRadians, Real> min_reading (0,150);
   
   for(int i = 0;i< sectorLbl_to_sectorData[sector_lbl].readings.size();i++){
      if(sectorLbl_to_sectorData[sector_lbl].readings[i].distance < min_reading.second){
         min_reading.second = sectorLbl_to_sectorData[sector_lbl].readings[i].distance;
         min_reading.first = sectorLbl_to_sectorData[sector_lbl].readings[i].angle;
      }
   }

   return min_reading;
}




std::array<int,4> CFootBotWall::extractFeatures(){
   int n_feature_interval = 8;
   CRadians aperture;
   aperture.FromValueInDegrees(360.0/n_feature_interval);
   CRadians start = -CRadians::PI + aperture/2;
   std::map<int,int> intervalID_to_nMin;
   std::vector<int> intervalID;
   std::array<int,4> feature = {0,0,0,0};
   bool placed;
   int d = 0;
   


   for(auto lm : lmr_new){
      placed = false;
      for(int i=0;i<n_feature_interval-1;i++){
         if(lm.angle > start + i*aperture && lm.angle <= start + (i+1)*aperture){
               intervalID_to_nMin[i] = 1;
               placed = true;
         }
      }

      if(!placed)
         intervalID_to_nMin[n_feature_interval-1] = 1;

   }

   for (const auto& [k,v] : intervalID_to_nMin){
      intervalID.push_back(k);
      std::cout << k;
   }
   std::cout << "\n";

   for(int i=0;i<intervalID.size()-1;i++){
      d = abs(intervalID[i+1] - intervalID[i]); //distanza in intervalli
      if(d > n_feature_interval/2)
         d = n_feature_interval - d;

      feature[d-1]++;

   }

   //handling the last - first elements
   d = abs(intervalID[intervalID.size()-1] - intervalID[0]);
   if(d > n_feature_interval/2)
      d = n_feature_interval - d;

   feature[d-1]++;


   return feature;
      
}


char CFootBotWall::predict(std::array<int,4> feature){
   
   char class_label = ' ';
   Real euc_distance = 0.0;
   Real min_euc_distace = 1000.0;

   for(const auto& [k,v] : classLbl_to_template){
      euc_distance = EucDistance(feature, v);
      if(euc_distance <= min_euc_distace){
         min_euc_distace = euc_distance;
         class_label = k;
      }

   }

   return class_label;
}


std::array<Real,2> CFootBotWall::StructuredExploration(
   Real r_distance_d,
   CRadians r_orientation_d,
   const CCI_RangeAndBearingSensor::TReadings& rab_readings){

   Real v, w; // linear velocity (x_component) & angular velocity (z_component)
   
   //Wall Following-----------------------------------------
   Real v_wf, w_wf, w_dis, w_ori;
   Real distance_error, orientation_error;
   Real L = 14.0; // wheels distance [cm] 

   struct angle_data min = {CRadians::ZERO, 150.0, 0, false};
   for(auto r : sectorLbl_to_sectorData['R'].readings){
      if (!r.occluded){
         if (r.distance <= min.distance){
            min.distance = r.distance;
            min.angle = r.angle;
         }  
      }
   }

   this->free_min = min;

   // angular velocity wf proportional to distance error + orientation error
   w_wf = 0.01*(r_distance_d - min.distance) + (-(r_orientation_d - min.angle).SignedNormalize().GetValue());

   // linear velocity wf base + linearly ramping down to 0 component (Arrival)
   v_wf = 2.0 + 5.0*(getMinReading('F').second - r_distance_d)/(150.0 - r_distance_d);


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


   //Combination
   v = v_wf;
   w = w_wf;

   return {v - w*L/2, v + w*L/2};

}



std::array<Real,2> CFootBotWall::UnstructuredExploration(
   const CCI_FootBotProximitySensor::TReadings& proximity_readings){

   /* Sum them together */
   Real v_l, v_r;
   CVector2 cAccumulator;

   for(int i = 0; i < proximity_readings.size(); ++i) 
      cAccumulator += CVector2(proximity_readings[i].Value, proximity_readings[i].Angle);
   
   cAccumulator /= proximity_readings.size();
   

   if (cAccumulator.Length() == 0.0){
      if(tic % 30 == 0 || chosen){
         
         if (!chosen){
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::uniform_real_distribution<float> distr(0, 1);
            
            if (distr(eng) >= 0.5)
               choice = 0;
            else
               choice = 1;

            chosen = true;
         }

         if (counter < 10){
            if (choice == 0){
               v_l = 2.0;
               v_r = -2.0;
            }
            else{
               v_l = -2.0;
               v_r = 2.0;
            }
         counter++;
         }
         
         else{
            counter = 0;
            chosen = false;
         }
         
      }
      else{
         v_l = 5.0;
         v_r = 5.0;
      }
   }
   else{
      if(m_cGoStraightAngleRange.WithinMinBoundIncludedMaxBoundIncluded(cAccumulator.Angle())) {
         if(cAccumulator.Angle().GetValue() > 0.0f) {
            v_l = 2.0;
            v_r = 0.0;
         }
         else {
            v_l = 0.0;
            v_r = 2.0;
         }
      }
      else{
         v_l = 5.0;
         v_r = 5.0;
      }
   }


   return {v_l, v_r};

}



void CFootBotWall::ControlStep() {

   // reset sectors_data (for the next control step)
   for (const auto& [sectorLbl, sectorData] : sectorLbl_to_sectorData)
      sectorLbl_to_sectorData[sectorLbl].readings.clear();
   pr.clear();
   
   // get the current step readings
   const CCI_FootBotDistanceScannerSensor::TReadingsMap& long_readings = m_pcDistanceS->GetLongReadingsMap();
   const CCI_FootBotDistanceScannerSensor::TReadingsMap& short_readings = m_pcDistanceS->GetShortReadingsMap();
   const CCI_RangeAndBearingSensor::TReadings& rab_readings = m_pcRangeAndBearingS->GetReadings();
   const CCI_PositioningSensor::SReading& robot_state = m_pcPositioning->GetReading();
   const CCI_FootBotProximitySensor::TReadings& proximity_readings = m_pcProximity->GetReadings();

   
   // add the current step readings in the map world_model_long (it starts empty then it grows then it stops)
   Real mod_distance;
   CVector2 rab_xy, shift_xy;
   CRadians start, end, delta;
   Real fb_radius = 12.0f;
   for (const auto& [angle, distance] : long_readings){
      mod_distance = distance;
      if (mod_distance == -1) mod_distance = 20.0f;
      if (mod_distance == -2) mod_distance = 150.0f;

      world_model_long[angle] = {angle, mod_distance, tic, false};
   }

   // individuazione dei raggi occlusi da altri robot 
   for (auto rr : rab_readings){
      if(rr.Range < 150.0f){
         rab_xy.FromPolarCoordinates(rr.Range, rr.HorizontalBearing);
         shift_xy.Set(rab_xy.GetX(), rab_xy.GetY());
         shift_xy = fb_radius * (shift_xy.Normalize().Perpendicularize());

         delta = (rab_xy + shift_xy).Angle() - rab_xy.Angle(); 

         for (const auto& [angle, data] : world_model_long)
            if ((angle - rab_xy.Angle()).SignedNormalize().GetAbsoluteValue() <= delta.SignedNormalize().GetAbsoluteValue())
               world_model_long[angle].occluded = true;
         
      }
   }

   for (const auto& [angle, distance] : short_readings){
      mod_distance = distance;
      if (mod_distance == -1) mod_distance = 4.0f;
      if (mod_distance == -2) mod_distance = 30.0f;

      world_model_short[angle] = {angle,mod_distance,tic};
   }


   // add the readings to the opportune sector based on the sector angle_interval
   for (const auto& [angle, angleData] : world_model_long){
      for (const auto& [sectorLbl, sectorData] : sectorLbl_to_sectorData){
         if (angle >= sectorData.angle_interval[0] && angle <= sectorData.angle_interval[1]){
            sectorLbl_to_sectorData[sectorLbl].readings.push_back(angleData);
         }
      }
   }


   processReadings('H');


   tic++;


   if(tic%10 == 0){
      // std::cout << "STORE\n";
      // std::cout << tic << "\n";
      // std::cout << dataset_step_data.size() << "\n";
      // std::cout << world_model_long.size() << "\n";
      // lmr_new.clear();
      // lMr.clear(); 
      // getLocalMinMaxReadings();


      // std::cout << "CANCELLA\n";
      // world_model_long.clear();
      // world_model_short.clear();
   
   }

   


   // compute the inputs
   auto input = StructuredExploration(35.0, -CRadians::PI_OVER_TWO, rab_readings);

   // Real INTERWHEEL_DISTANCE = 0.14f*100;

   // std::cout << "v: " << (input[0]+input[1]) * 0.5 << "\n";
   // std::cout << "w: " << (input[1]-input[0]) * 1/INTERWHEEL_DISTANCE << "\n";
      
   // set the inputs
   m_pcWheels->SetLinearVelocity(input[0], input[1]);
   

   // store step_data in step_data_dataset
   CRadians x,y,theta;
   robot_state.Orientation.ToEulerAngles(theta,y,x);

   if(GetId() == "fb_0"){
      dataset_step_data.push_back({tic,
                                   robot_state.Position.GetX(),
                                   robot_state.Position.GetY(),
                                   theta.GetValue(),
                                   input[0],
                                   input[1],
                                   world_model_long,
                                   world_model_short});
   }



   //dump the dataset into a .csv
   if(tic%100 == 0 && GetId() == "fb_0"){

      std::cout << "DUMPED\n";
      
      std::ofstream file;
      file.open("test_unstructured_1.csv", std::ios_base::app);
      for(auto step : dataset_step_data){

         if (step.long_readings.size() == 120 && step.short_readings.size() == 120){
            std::string row = "";
            row += std::to_string(step.clock)+"|";
            row += std::to_string(step.x)+"|";
            row += std::to_string(step.y)+"|";
            row += std::to_string(step.theta)+"|";
            row += std::to_string(step.v_left)+"|";
            row += std::to_string(step.v_right)+"|";

            for(const auto& [angle, angle_data] : step.long_readings){
               row += std::to_string(angle_data.angle.GetValue())+"|";
               row += std::to_string(angle_data.distance)+"|";
            }

            for(const auto& [angle, angle_data] : step.short_readings){
               row += std::to_string(angle_data.angle.GetValue())+"|";
               row += std::to_string(angle_data.distance)+"|";
            }

            row += "\n";

            //file << row;
         }
      }

      file.close();

      dataset_step_data.clear();
   }

  

}







REGISTER_CONTROLLER(CFootBotWall, "footbot_wall_controller")
