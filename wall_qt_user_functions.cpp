#include "wall_qt_user_functions.h"


/****************************************/
/****************************************/

CWALLQTUserFunctions::CWALLQTUserFunctions() {
   RegisterUserFunction<CWALLQTUserFunctions,CFootBotEntity>(&CWALLQTUserFunctions::Draw);
}

/****************************************/
/****************************************/



void CWALLQTUserFunctions::Draw(CFootBotEntity& c_entity) {

   DrawText(CVector3(0.0, 0.0, 0.5), c_entity.GetId().c_str());


   
   controller = dynamic_cast<CFootBotWall&> (c_entity.GetControllableEntity().GetController());

   auto min_angle = controller.free_min.angle;
   auto min_length = controller.free_min.distance;
   CVector2 end = CVector2(min_length, min_angle);
   CRay3 ray = CRay3(CVector3(0.0, 0.0, 0.01) ,CVector3(end.GetX()*0.01, (end.GetY())*0.01, 0.01));
   DrawRay(ray, CColor::BLACK, 2.0f);

   ray = CRay3(CVector3(0.0, 0.0, 2.01) ,CVector3(controller.nearest_robot_xy.GetX()*0.01, (controller.nearest_robot_xy.GetY())*0.01, 2.01));
   DrawRay(ray, CColor::RED, 2.0f);


   if(controller.GetId() == "fb_13"){
   
      // for (auto p : controller.pr){
      //    CVector2 end = CVector2(p.distance, p.angle);
      //    CRay3 ray = CRay3(CVector3(0.0, 0.0, 0.01) ,CVector3(end.GetX()*0.01, (end.GetY())*0.01, 0.01));
      //    DrawRay(ray);
      //    //DrawText(CVector3(end.GetX()*0.01, (end.GetY())*0.01, 0.01), std::to_string(p.distance));
      // }
      

      // for (auto r : controller.sectorLbl_to_sectorData['R'].readings){
      //    CVector2 end = CVector2(r.distance, r.angle);
      //    CRay3 ray = CRay3(CVector3(0.0, 0.0, 0.01) ,CVector3(end.GetX()*0.01, (end.GetY())*0.01, 0.01));
      //    if(r.occluded)
      //       DrawRay(ray, CColor::BLUE, 2.0f);
      //    else
      //       DrawRay(ray);
      //    DrawText(CVector3(end.GetX()*0.01, (end.GetY())*0.01, 0.01), std::to_string(r.age));

      // }
   }
}


void CWALLQTUserFunctions::DrawInWorld() {
   // Real x = 0.85;
   // Real y = 0.0;
   // Real dx = 0.03;
   // Real l;
   // bool in;
   
   // for (auto p : controller.pr){
   //    in = false;
   //    CVector2 end = CVector2(p.distance, p.angle);
   //    l = end.Length();
   //    CRay3 ray = CRay3(CVector3(x, y, 0.01), CVector3(x, y+l*0.01, 0.01));
   //    for(auto min : controller.lmr_new)
   //       if(p.angle == min.angle)
   //          in = true;
   //    if (in)
   //       DrawRay(ray, CColor::BLACK, 2.0f);
   //    else
   //       DrawRay(ray);
      
   //    x = x+dx;
   // }


   // for(auto p : controller.position_data_map){
   //    if (p.predicted_class_lbl == 'V')
   //       DrawPoint(p.coordinates, CColor::RED, 10.0);
   //    else if (p.predicted_class_lbl == 'C')
   //       DrawPoint(p.coordinates, CColor::YELLOW, 10.0);
   //    else if (p.predicted_class_lbl == 'G')
   //       DrawPoint(p.coordinates, CColor::GREEN, 10.0);
   //    else if (p.predicted_class_lbl == 'I')
   //       DrawPoint(p.coordinates, CColor::BLUE, 10.0);

   //    //DrawText(p.coordinates, std::to_string(p.true_class_lbl));

   // }
}

/****************************************/
/****************************************/

REGISTER_QTOPENGL_USER_FUNCTIONS(CWALLQTUserFunctions, "wall_qt_user_functions")
