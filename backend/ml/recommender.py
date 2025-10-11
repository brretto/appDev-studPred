# backend/ml/recommender.py

def get_recommendation(data, prediction):
    """
    Generates advice based on student data and prediction result.
    This version safely handles missing data.
    """
    # Safely get values using .get(key, default_value)
    study_hours = data.get('StudyHoursPerWeek')
    attendance = data.get('AttendanceRate')
    grade = data.get('PreviousGrade')

    if prediction == 'FAIL':
        # Convert to float for comparison, handling cases where it might be None
        try:
            if study_hours and float(study_hours) < 10:
                return "Your study hours are low. Try creating a consistent study schedule of at least 10 hours per week to improve your grasp of the material."
            if attendance and float(attendance) < 80:
                return "Low attendance can impact your grades. Make an effort to attend all classes to stay on track."
        except (ValueError, TypeError):
            # This handles cases where the data isn't a number
            pass 
        return "Focus on reviewing your notes from previous lessons and don't hesitate to ask your teachers for help on topics you find difficult."
    
    if prediction == 'PASS':
        try:
            if grade and float(grade) < 85:
                return "Great job passing! To aim for higher marks, try dedicating a little more time to your most challenging subjects."
        except (ValueError, TypeError):
            pass
        return "You are doing great! Keep up the hard work and continue to challenge yourself."
    
    return "No specific recommendation available."