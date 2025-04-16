import os
import re
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, Mapping
from argparse import Namespace
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Param, Metric
import yaml

class MLflowLogger:
    """MLflow logger for YOLOv7 training experiments.
    
    Args:
        experiment_name (str): Name of the experiment
        run_name (str): Name of the run
        tracking_uri (str): MLflow tracking URI
        tags (Dict[str, Any]): Tags for the run
        save_dir (str): Directory to save MLflow artifacts
        log_model (bool): Whether to log model checkpoints
        artifact_location (str): Location for artifacts
        run_id (str): Existing run ID to resume
        synchronous (bool): Whether to log synchronously
    """
    
    def __init__(
        self,
        opt,
        data_dict,
        experiment_name: str = "yolov7",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        save_dir: str = "./mlruns",
        log_model: bool = True,
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        synchronous: bool = False,
        current_epoch: int = 0
    ):
        self.current_epoch = current_epoch
        self.opt = opt
        self.data_dict = data_dict
        self._experiment_name = experiment_name
        self._run_name = run_name or Path(opt.save_dir).name
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", f"file:{save_dir}")
        self.tags = tags or {}
        self._log_model = log_model
        self._prefix = ""
        self._artifact_location = artifact_location
        self._run_id = run_id
        self._initialized = False
        self._logged_model_time = {}
        self._checkpoint_callback = None

        # Initialize MLflow client
        self._mlflow_client = MlflowClient(self._tracking_uri)
        self._setup_experiment()

        # Log initial configuration
        self.log_hyperparameters(vars(opt))
        self.log_hyperparameters(opt.hyp)
        self.log_artifact(opt.data)
        self.log_artifact(opt.cfg)

    def _setup_experiment(self):
        """Set up MLflow experiment and run."""
        # Get or create experiment
        experiment = self._mlflow_client.get_experiment_by_name(self._experiment_name)
        if experiment:
            self._experiment_id = experiment.experiment_id
        else:
            self._experiment_id = self._mlflow_client.create_experiment(
                name=self._experiment_name,
                artifact_location=self._artifact_location
            )

        # Start run
        if self._run_id:
            self._run = self._mlflow_client.get_run(self._run_id)
        else:
            self._run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id,
                tags={**self.tags, "mlflow.runName": self._run_name}
            )
            self._run_id = self._run.info.run_id

        mlflow.start_run(
            run_id=self._run_id,
            experiment_id=self._experiment_id,
            log_system_metrics=True
        )
        self._initialized = True

    @property
    def experiment(self) -> MlflowClient:
        """Get MLflow client."""
        return self._mlflow_client

    def log_hyperparameters(self, params: Union[Dict[str, Any], Namespace]):
        """Log hyperparameters to MLflow."""
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        
        # Log in batches of 100 parameters
        params_list = [Param(key=k, value=str(v)[:250]) for k, v in params.items()]
        for i in range(0, len(params_list), 100):
            self.experiment.log_batch(
                run_id=self._run_id,
                params=params_list[i:i+100]
            )

    def log(self, metrics: Mapping[str, float]):
        """Log metrics to MLflow."""
        metrics = self._add_prefix(metrics)
        timestamp = int(time.time() * 1000)
        
        metrics_list = []
        for k, v in metrics.items():
            k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)  # Sanitize metric name
            metrics_list.append(Metric(
                key=k,
                value=v,
                timestamp=timestamp,
                step=self.current_epoch
            ))
            
        self.experiment.log_batch(
            run_id=self._run_id,
            metrics=metrics_list
        )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, weights_dir: Path, opt, epoch, fitness, best_model=False):
        """Log model checkpoint to MLflow."""
        if not self._log_model:
            return
        
        run_dir = weights_dir.parent
        if best_model:
            self.log_artifact(str(weights_dir / "best.pt"))
        
        artifacts = {
            "model": str(weights_dir / "last.pt"),
            "config": str(run_dir / "opt.yaml"),
            "hyp": str(run_dir / "hyp.yaml")
        }
        
        # Log model artifacts
        for name, path in artifacts.items():
            if Path(path).exists():
                self.log_artifact(path)
        
    
    def end_epoch(self, best_result=False):
        self.current_epoch += 1

        # Optionally, set a tag for the epoch if this is the best result so far
        if best_result:
            mlflow.set_tag("best_model_epoch", self.current_epoch)

        # Increment the epoch counter
        self.current_epoch += 1

    def finish_run(self, status: str = "FINISHED"):
        """Finalize the run."""
        if self._initialized:
            self.experiment.set_terminated(self._run_id, status)
            mlflow.end_run()

    def _convert_params(self, params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        if isinstance(params, Namespace):
            params = vars(params)
        return params

    def _flatten_dict(self, params: Dict[str, Any], delimiter: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        def _flatten(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{delimiter}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        return _flatten(params)

    def _add_prefix(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        """Add prefix to metrics."""
        return {f"{self._prefix}{k}": v for k, v in metrics.items()}

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self._run_id

    @property
    def experiment_id(self) -> str:
        """Get current experiment ID."""
        return self._experiment_id

    def __del__(self):
        """Destructor to ensure clean shutdown."""
        if self._initialized:
            self.finish_run()