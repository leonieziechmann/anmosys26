import random

class ThermalDampenerEnv:
    def __init__(self, initial_kappa: float = 12.0, initial_temp: float = 120.0):
        self.kappa = initial_kappa
        self.temperature = initial_temp
        
    def step(self, delta_kappa: float):
        # Apply adjustment to Kappa
        self.kappa += delta_kappa
        # Prevent kappa from going negative
        self.kappa = max(0.0, self.kappa)
        
        # Equilibrium temperature for current kappa:
        # If kappa = 5.0, T_eq = 25.0 (PERFECT)
        # If kappa is high, T_eq is high (BOILING)
        # If kappa is low, T_eq is low (FREEZING)
        t_eq = 25.0 + 15.0 * (self.kappa - 5.0)
        
        # Temperature moves towards equilibrium with some inertia
        inertia = 0.6
        noise = random.uniform(-1.0, 1.0)
        self.temperature = self.temperature + inertia * (t_eq - self.temperature) + noise
        
        return self.temperature, self.kappa

    def get_status_log(self) -> str:
        return f"Current Temperature: {self.temperature:.2f}K (Kappa: {self.kappa:.2f})"
