import math
import scipy.integrate as integrate
from config import CURRICULUM_BASE_NOISE, CURRICULUM_SIGMA_1, CURRICULUM_SIGMA_2

def get_vph(time_hours: float, total_daily_cars: int) -> float:
    """
    Returns the target traffic volume (vehicles per hour) for a given time of day.
    """
    def shape(t):
        # 8 AM peak (morning rush)
        peak1 = math.exp(-((t - 8.0) ** 2) / (2 * CURRICULUM_SIGMA_1 ** 2))
        # 4 PM peak (evening rush), slightly wider
        peak2 = math.exp(-((t - 16.0) ** 2) / (2 * CURRICULUM_SIGMA_2 ** 2))
        # Baseline quiet traffic (night time)
        return peak1 + peak2 + CURRICULUM_BASE_NOISE

    # Numerically integrate the shape function from hour 0 to hour 24
    # 'area' represents the unscaled total "number of cars" under the basic shape
    area, _ = integrate.quad(shape, 0, 24)
    
    # Calculate the scaling factor needed to reach our exact total_daily_cars
    scale_factor = total_daily_cars / area
    
    # Multiply the shape at the specific time by the scaling factor
    vph = shape(time_hours) * scale_factor
    
    return vph
