#ifndef WALL_QTUSER_FUNCTIONS_H
#define WALL_QTUSER_FUNCTIONS_H

#include <argos3/plugins/simulator/visualizations/qt-opengl/qtopengl_user_functions.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include "footbot_wall.h"



using namespace argos;

class CWALLQTUserFunctions : public CQTOpenGLUserFunctions {

public:

   CFootBotWall controller;

   CWALLQTUserFunctions();

   virtual ~CWALLQTUserFunctions() {}

   void Draw(CFootBotEntity& c_entity);

   void DrawInWorld();
   
};

#endif
