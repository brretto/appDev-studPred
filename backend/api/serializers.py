from rest_framework import serializers
from .models import SurveyResponse, Prediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['predicted_result', 'confidence_score', 'recommendation']

class SurveyResponseSerializer(serializers.ModelSerializer):
    # This nested serializer includes the prediction data with each response
    prediction = PredictionSerializer(read_only=True)

    class Meta:
        model = SurveyResponse
        # List all the fields you want to send to the frontend
        fields = [
            'id', 'gender', 'attendance_rate', 'study_hours_per_week',
            'previous_grade', 'extracurricular_activities', 'parental_support',
            'online_classes_taken', 'created_at', 'prediction'
        ]