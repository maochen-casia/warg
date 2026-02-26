import torch

def get_metric(metric_name, sat_image_size=None, sat_image_coverage=None):    
    if metric_name == 'mean':
        return MeanMetric()
    elif metric_name == 'std':
        return StdMetric()
    elif metric_name == 'median':
        return QuantileMetric(0.5)
    elif metric_name == 'boxplot':
        return BoxPlotMetric()
    elif 'quantile' in metric_name:
        q = float(metric_name.split('_')[1])
        return QuantileMetric(q)
    elif 'accuracy' in metric_name:
        assert sat_image_size and sat_image_coverage, "sat_image_size and sat_image_coverage must be provided for accuracy metric."
        ppm = sat_image_size / sat_image_coverage  # pixels per meter
        pixel_thresh = float(metric_name.split('_')[1]) # accuracy_10 -> 10 pixels threshold
        return AccuracyMetric(ppm, pixel_thresh)

def translation_error(pred, label):
    t_left2world_pred = pred['t_left2world']
    t_left2world_label = label['t_left2world'].to(t_left2world_pred.device)
    error = torch.norm(t_left2world_pred - t_left2world_label, dim=-1)
    return error

class BaseMetric:

    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, pred, label):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def aggregate(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
class MeanMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        self.total += error.sum().item()
        self.count += error.shape[0]
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return self.total / self.count

class StdMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.errors = torch.empty(0)
        self.count = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return torch.std(self.errors)

class QuantileMetric(BaseMetric):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def reset(self):
        self.errors = torch.empty(0)
        self.count = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return torch.quantile(self.errors, self.quantile)

class AccuracyMetric(BaseMetric):
    def __init__(self, ppm, pixel_thresh):
        super().__init__()
        self.meter_thresh = pixel_thresh / ppm
    
    def reset(self):
        self.correct = 0
        self.total = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        mask = error < self.meter_thresh
        self.correct = self.correct + torch.sum(mask).item()
        self.total = self.total + error.shape[0]
        return error

    def aggregate(self):
        if self.total == 0:
            return 0
        return self.correct / self.total

class BoxPlotMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.errors = torch.empty(0)
        self.count = 0

    def reset(self):
        self.errors = torch.empty(0)
        self.count = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        return error

    def aggregate(self):
        if self.count == 0:
            return {
                "median": 0,
                "q1": 0,
                "q3": 0,
                "min": 0,
                "max": 0,
                "outliers": torch.empty(0)
            }

        # Calculate the key statistics
        sorted_errors = torch.sort(self.errors).values
        q1 = torch.quantile(sorted_errors, 0.25)
        q3 = torch.quantile(sorted_errors, 0.75)
        median = torch.quantile(sorted_errors, 0.5)
        iqr = q3 - q1
        
        # Calculate the whiskers (max and min within 1.5 * IQR)
        min_value = sorted_errors[sorted_errors >= (q1 - 1.5 * iqr)].min()
        max_value = sorted_errors[sorted_errors <= (q3 + 1.5 * iqr)].max()

        # Detect outliers (values outside the whiskers)
        outliers = sorted_errors[(sorted_errors < (q1 - 1.5 * iqr)) | (sorted_errors > (q3 + 1.5 * iqr))]

        return {
            "median": median.item(),
            "q1": q1.item(),
            "q3": q3.item(),
            "min": min_value.item(),
            "max": max_value.item(),
            "outliers": outliers.tolist()
        }