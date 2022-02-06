#ifndef FOOTBOT_WALL_H
#define FOOTBOT_WALL_H

/*
 * Include some necessary headers.
 */

#include <argos3/core/control_interface/ci_controller.h>

// actuators
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_actuator.h>
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_distance_scanner_actuator.h>

//sensors
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_distance_scanner_sensor.h>
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_proximity_sensor.h>
#include <argos3/plugins/robots/generic/control_interface/ci_range_and_bearing_sensor.h>
#include <argos3/plugins/robots/generic/control_interface/ci_positioning_sensor.h>

#include <experimental/random>

/* 2D vector definition */
#include <argos3/core/utility/math/vector2.h>

#include <fstream>


using namespace argos;


class CFootBotWall : public CCI_Controller {

private:


   CCI_DifferentialSteeringActuator* m_pcWheels;
   CCI_FootBotDistanceScannerActuator* m_pcDistanceA;
   CCI_FootBotDistanceScannerSensor* m_pcDistanceS;
   CCI_FootBotProximitySensor* m_pcProximity;
   CCI_RangeAndBearingSensor* m_pcRangeAndBearingS;
   CCI_PositioningSensor* m_pcPositioning;

   CDegrees m_cAlpha;
   Real m_fDelta;
   Real m_fWheelVelocity;
   CRange<CRadians> m_cGoStraightAngleRange;


   CRadians chosen_direction;
   bool chosen = false;
   int choice;
   int counter = 0;



public:

   int tic = 0;

   struct angle_data{
      CRadians angle;
      Real distance;
      int age;
      bool occluded;
   };

   std::map<CRadians, struct angle_data> world_model_short;
   std::map<CRadians, struct angle_data> world_model_long;


   std::vector<struct angle_data> lmr_new;
   std::vector<struct angle_data> lMr; 
   std::vector<struct angle_data> pr;
   struct angle_data free_min;
   CVector2 nearest_robot_xy;
 

   struct step_data{
      int clock;
      Real x;
      Real y;
      Real theta;
      Real v_right;
      Real v_left;
      std::map<CRadians, struct angle_data> long_readings;
      std::map<CRadians, struct angle_data> short_readings;
   };

   std::vector<struct step_data> dataset_step_data;

   struct sector_data{
      std::array<CRadians,2> angle_interval;
      std::vector<struct angle_data> readings;

   } R, F, H;

   std::map<char,struct sector_data> sectorLbl_to_sectorData;

   std::map<char,std::array<int,4>> classLbl_to_template;

   void processReadings(char sector_lbl);
   std::pair<CRadians,Real> getMinReading(char sector_lbl);
   void getLocalMinMaxReadings();

   std::array<int,4> extractFeatures();
   Real EucDistance(std::array<int,4> u, std::array<int,4> v);
   char predict(std::array<int,4> features);
   
   std::array<Real,2> StructuredExploration(
      Real r_distance_d, 
      CRadians r_orientation_d, 
      const CCI_RangeAndBearingSensor::TReadings&  rab_readings);
   
   std::array<Real,2> UnstructuredExploration(
      const CCI_FootBotProximitySensor::TReadings& proximity_readings);

   /* Class constructor. */
   CFootBotWall();

   /* Class destructor. */
   virtual ~CFootBotWall() {}

   /*
    * This function initializes the controller.
    * The 't_node' variable points to the <parameters> section in the XML
    * file in the <controllers><footbot_diffusion_controller> section.
    */
   virtual void Init(TConfigurationNode& t_node);

   /*
    * This function is called once every time step.
    * The length of the time step is set in the XML file.
    */
   virtual void ControlStep();

   /*
    * This function resets the controller to its state right after the
    * Init().
    * It is called when you press the reset button in the GUI.
    * In this example controller there is no need for resetting anything,
    * so the function could have been omitted. It's here just for
    * completeness.
    */
   virtual void Reset() {}

   /*
    * Called to cleanup what done by Init() when the experiment finishes.
    * In this example controller there is no need for clean anything up,
    * so the function could have been omitted. It's here just for
    * completeness.
    */
   virtual void Destroy() {}


   




};

#endif
