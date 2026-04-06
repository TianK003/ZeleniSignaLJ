import math
import scipy.integrate as integrate

def get_vph(time_hours: float, total_daily_cars: int) -> float:
    """
    Calculates the exact Vehicles Per Hour (vph) at a given time of day.
    Uses two overlapping bell curves (Gaussian functions) plus a small baseline.
    Peak 1: 8:00 AM
    Peak 2: 16:00 (4:00 PM)
    
    The curve is automatically scaled so that the total area underneath 
    equals 'total_daily_cars'.
    """
    
    def shape(t):
        # 8 AM peak (morning rush)
        peak1 = math.exp(-((t - 8.0) ** 2) / (2 * 1.5 ** 2))
        # 4 PM peak (evening rush), slightly wider
        peak2 = math.exp(-((t - 16.0) ** 2) / (2 * 2.0 ** 2))
        # 0.1 is the baseline quiet traffic (night time)
        return peak1 + peak2 + 0.1

    # Numerically integrate the shape function from hour 0 to hour 24
    # 'area' represents the unscaled total "number of cars" under the basic shape
    area, _ = integrate.quad(shape, 0, 24)
    
    # Calculate the scaling factor needed to reach our exact total_daily_cars
    scale_factor = total_daily_cars / area
    
    # Multiply the shape at the specific time by the scaling factor
    vph = shape(time_hours) * scale_factor
    
    return vph
