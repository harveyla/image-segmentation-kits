__all__ = ['error_rate', 'accuracy', 'accuracy_thresh', 'dice', 'exp_rmspe', 'fbeta','FBeta', 'mse', 'mean_squared_error',
            'mae', 'mean_absolute_error', 'rmse', 'root_mean_squared_error', 'msle', 'mean_squared_logarithmic_error',
            'explained_variance', 'r2_score', 'top_k_accuracy', 'KappaScore', 'ConfusionMatrix', 'MatthewsCorreff',
            'Precision', 'Recall', 'Seg_Precision', 'Seg_Recall', 'Seg_F1', 'R2Score', 'ExplainedVariance', 'ExpRMSPE', 'RMSE', 'Perplexity']

class Segmentation(Callback):
   
    def on_epoch_begin(self, **kwargs):
        self.tp, self.fp, self.fn = 0, 0, 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        pred = last_output
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        pred = pred.cpu()
        last_target = last_target.cpu()
        self.tp += ((pred==1) * (last_target==1)).float().sum()
        self.fp += ((pred==1) * (last_target==0)).float().sum()
        self.fn += ((pred==0) * (last_target==1)).float().sum()
    
class Seg_Precision(Segmentation):
    def on_epoch_end(self, last_metrics, **kwargs):
        self.precision = self.tp/(self.tp+self.fp)
        return add_metrics(last_metrics, self.precision)
class Seg_Recall(Segmentation):
    def on_epoch_end(self, last_metrics, **kwargs):
        self.recall = self.tp/(self.tp+self.fn)
        return add_metrics(last_metrics, self.recall)
class Seg_F1(Segmentation):
    def on_epoch_end(self, last_metrics, **kwargs):
        self.precision = self.tp/(self.tp+self.fp)
        self.recall = self.tp/(self.tp+self.fn)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return add_metrics(last_metrics, self.f1)
