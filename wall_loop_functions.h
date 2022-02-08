#ifndef WALL_LOOP_FUNCTIONS_H
#define WALL_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
//#include "footbot_wall.h"

using namespace argos;

class CWallLoopFunctions : public CLoopFunctions {
   
public:

   CFootBotEntity pcFB;

   virtual ~CWallLoopFunctions() {}

   virtual void Init(TConfigurationNode& t_tree);

   virtual void PostStep();


};

#endif
