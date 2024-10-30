from modules.models import OslModel, OsltModel, PriorityModel, TypeModel, UserGradeModel

class CategorizerInference:
    def __init__(self, model_path):
        self.priority_model = PriorityModel(model_path)
        self.type_model = TypeModel(model_path)
        self.user_grade_model = UserGradeModel(model_path)
        self.osl_model = OslModel(model_path)
        self.oslt_model = OsltModel(model_path)
        
    def infer(self, ticket_text, partner_name=None):
        priority = self.priority_model.predict(ticket_text)
        ticket_type = self.type_model.predict(ticket_text)
        user_grade = self.user_grade_model.predict(ticket_text)
        osl = self.osl_model.predict(ticket_text, partner_name)
        oslt = self.oslt_model.predict(ticket_text, partner_name)
        
        json_output = {
            "ticket_type": ticket_type,
            "priority": priority,
            "osl": osl,
            "oslt": oslt,
            "user_grade": user_grade
        }

        return json_output