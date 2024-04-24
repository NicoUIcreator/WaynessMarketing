def calculate_wep(heart_rate, duration):
                if 70 <= heart_rate <= 89:
                    wep = duration * 4  
                elif 90 <= heart_rate <= 109:
                    wep = duration * 5
                elif 110 <= heart_rate <= 129:
                    wep = duration * 6
                elif heart_rate >= 130:
                    wep = duration * 7
                else:
                    wep = 0  
                return wep

def categorize_activity(heart_rate):

    if heart_rate <= 79:
        return 0  
    elif 80 <= heart_rate <= 99:
        return 5
    elif 100 <= heart_rate <= 120:
        return 7 
    else:
        return 10 