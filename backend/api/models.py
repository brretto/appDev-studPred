from django.db import models

# Create your models here.

class SurveyResponse(models.Model):
    # Student's input data
    gender = models.CharField(max_length=10)
    attendance_rate = models.FloatField()
    study_hours_per_week = models.FloatField()
    previous_grade = models.FloatField()
    extracurricular_activities = models.BooleanField(default=False)
    parental_support = models.CharField(max_length=10)
    online_classes_taken = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Response {self.id}"

class Prediction(models.Model):
    # Link to the original survey response
    survey_response = models.OneToOneField(SurveyResponse, on_delete=models.CASCADE, related_name='prediction')
    
    # The model's output
    predicted_result = models.CharField(max_length=4) # 'PASS' or 'FAIL'
    confidence_score = models.FloatField()
    recommendation = models.TextField(blank=True, null=True)
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_result} ({self.confidence_score:.2%})"