#include "wall_loop_functions.h"
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>


void CWallLoopFunctions::Init(TConfigurationNode& t_tree){

    // CSpace::TMapPerType& tFBMap = GetSpace().GetEntitiesByType("foot-bot");
    
    // for(CSpace::TMapPerType::iterator it = tFBMap.begin();it != tFBMap.end();++it)
    //     pcFB = any_cast<CFootBotEntity*>(it->second);

    // pcFB = dynamic_cast<CFootBotEntity&> (GetSpace().GetEntity("fb_0"));

    CFootBotEntity* pcFB;
//    for(int i=0;i<5;i++){
//        pcFB = new CFootBotEntity("fb_"+std::to_string(i), "fwc", CVector3(-i/3.0-2.0,2.3,0), CQuaternion(CRadians::ZERO, CVector3(0,0,1)));
//        AddEntity(*pcFB);
//    }
//
//    for(int i=0;i<5;i++){
//        pcFB = new CFootBotEntity("fb_1"+std::to_string(i), "fwc", CVector3(i/3.0+2.0,3.2,0), CQuaternion(-CRadians::PI, CVector3(0,0,1)));
//        AddEntity(*pcFB);
//    }

    for(int i=0;i<20;i++){
        pcFB = new CFootBotEntity("fb_"+std::to_string(i), "fwc", CVector3(0.3,i/3.0,0), CQuaternion(CRadians::PI_OVER_TWO, CVector3(0,0,1)));
        AddEntity(*pcFB);
    }
    
//    pcFB = new CFootBotEntity("fb_0", "fwc", CVector3(1.3,0,0), CQuaternion(CRadians::PI, CVector3(0,0,1)));
//    AddEntity(*pcFB);
//    pcFB = new CFootBotEntity("fb_1", "fwc", CVector3(0,-1,0), CQuaternion(CRadians::PI_OVER_TWO, CVector3(0,0,1)));
//    AddEntity(*pcFB);
}


void CWallLoopFunctions::PostStep() {


    // if (GetSpace().GetSimulationClock() % 10 == 0 && GetSpace().GetSimulationClock() != 0){

    //     //std::cout << "SPOSTA" << "\n";

    //     std::random_device rd;
    //     std::default_random_engine eng(rd());
    //     std::uniform_real_distribution<float> distr(0, 1);

    //     float x = (-4+0.35) + distr(eng)*(8-0.35*2);
    //     float y = (2+0.35) + distr(eng)*(1.5-0.35*2);
    //     //MoveEntity(pcFB.GetEmbodiedEntity(), CVector3(x,y,0), CQuaternion());

    // }


  
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CWallLoopFunctions, "wall_loop_functions")
