"""Command-line interface for GRPO Trainer."""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from grpo_trainer.config import Config
from grpo_trainer.trainer import GRPOTrainerWrapper, train_grpo
from grpo_trainer.evaluate import run_evaluation

app = typer.Typer(
    name="grpo-trainer",
    help="Advanced GRPO Training Framework for LLM Fine-tuning",
    add_completion=False,
)

console = Console()
logger = logging.getLogger(__name__)


def show_banner():
    """Display the GRPO Trainer banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                     GRPO Trainer v2.0                         ║
║     Advanced GRPO Training Framework for LLM Fine-tuning      ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def show_config_summary(config: Config):
    """Display configuration summary."""
    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", config.model.name)
    table.add_row("Dataset", config.data.name)
    table.add_row("Output Dir", config.training.output_dir)
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("Epochs", str(config.training.num_train_epochs))
    table.add_row("LoRA Enabled", str(config.lora.enabled))
    table.add_row("LoRA Rank", str(config.lora.r))
    
    console.print(table)


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model name or path (overrides config)",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset", "-d",
        help="Dataset name (gsm8k, math, svamp, etc.)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for model and logs",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name", "-n",
        help="Name for this training run",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Per-device training batch size",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of training epochs",
    ),
    max_steps: Optional[int] = typer.Option(
        None,
        "--max-steps",
        help="Maximum training steps (overrides epochs)",
    ),
    lora: Optional[bool] = typer.Option(
        None,
        "--lora/--no-lora",
        help="Enable/disable LoRA",
    ),
    lora_rank: Optional[int] = typer.Option(
        None,
        "--lora-rank", "-r",
        help="LoRA rank",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume from checkpoint path",
    ),
    report_to: Optional[str] = typer.Option(
        None,
        "--report-to",
        help="Reporting backend (wandb, tensorboard, none)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging",
    ),
):
    """Train a model using GRPO."""
    show_banner()
    
    # Load or create config
    if config_path and config_path.exists():
        console.print(f"[cyan]Loading config from:[/cyan] {config_path}")
        config = Config.from_yaml(config_path)
    else:
        console.print("[yellow]Using default configuration[/yellow]")
        config = Config()
    
    # Override with CLI arguments
    if model_name:
        config.model.name = model_name
    if dataset:
        config.data.name = dataset
    if output_dir:
        config.training.output_dir = output_dir
    if run_name:
        config.training.run_name = run_name
    if learning_rate:
        config.training.learning_rate = learning_rate
    if batch_size:
        config.training.per_device_train_batch_size = batch_size
    if epochs:
        config.training.num_train_epochs = epochs
    if max_steps:
        config.training.max_steps = max_steps
    if lora is not None:
        config.lora.enabled = lora
    if lora_rank:
        config.lora.r = lora_rank
    if resume:
        config.training.resume_from_checkpoint = resume
    if report_to:
        config.training.report_to = report_to
    if seed:
        config.training.seed = seed
    
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Show config summary
    show_config_summary(config)
    
    # Confirm before starting
    if not typer.confirm("\nStart training with this configuration?", default=True):
        console.print("[yellow]Training cancelled.[/yellow]")
        raise typer.Exit()
    
    # Run training
    console.print("\n[bold green]Starting training...[/bold green]\n")
    
    try:
        trainer = train_grpo(config)
        console.print("\n[bold green]✓ Training completed successfully![/bold green]")
        console.print(f"[cyan]Model saved to:[/cyan] {config.training.output_dir}")
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: str = typer.Argument(
        ...,
        help="Path to the trained model",
    ),
    dataset: str = typer.Option(
        "gsm8k",
        "--dataset", "-d",
        help="Dataset to evaluate on",
    ),
    split: str = typer.Option(
        "test",
        "--split", "-s",
        help="Dataset split to use",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples", "-n",
        help="Number of samples to evaluate",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size", "-b",
        help="Batch size for evaluation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """Evaluate a trained model."""
    show_banner()
    
    console.print(f"[cyan]Evaluating model:[/cyan] {model_path}")
    console.print(f"[cyan]Dataset:[/cyan] {dataset} ({split})")
    
    try:
        results = run_evaluation(
            model_path=model_path,
            dataset_name=dataset,
            split=split,
            num_samples=num_samples,
            batch_size=batch_size,
            output_file=str(output_file) if output_file else None,
            verbose=verbose,
        )
        
        # Display results
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in results.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
        
        if output_file:
            console.print(f"\n[cyan]Results saved to:[/cyan] {output_file}")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Evaluation failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def init_config(
    output_path: Path = typer.Argument(
        "config.yaml",
        help="Output path for configuration file",
    ),
    preset: str = typer.Option(
        "default",
        "--preset", "-p",
        help="Configuration preset (default, gsm8k, math, fast)",
    ),
):
    """Generate a configuration file."""
    show_banner()
    
    config = Config()
    
    # Apply presets
    if preset == "gsm8k":
        config.data.name = "gsm8k"
        config.data.use_one_shot = True
        config.training.run_name = "grpo-gsm8k"
    elif preset == "math":
        config.data.name = "math"
        config.data.use_one_shot = True
        config.training.run_name = "grpo-math"
        config.training.max_completion_length = 1024
    elif preset == "fast":
        config.training.num_train_epochs = 1
        config.training.max_steps = 100
        config.training.logging_steps = 10
        config.training.save_steps = 50
        config.data.max_samples = 500
    
    config.to_yaml(output_path)
    console.print(f"[green]✓ Configuration saved to:[/green] {output_path}")
    console.print("\n[cyan]Edit the file to customize your training configuration.[/cyan]")


@app.command()
def info():
    """Show package information and system details."""
    import torch
    import transformers
    import trl
    import peft
    
    show_banner()
    
    table = Table(title="System Information", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Version/Info", style="green")
    
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Transformers", transformers.__version__)
    table.add_row("TRL", trl.__version__)
    table.add_row("PEFT", peft.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            table.add_row(f"GPU {i}", f"{props.name} ({props.total_memory // 1024**3}GB)")
    
    console.print(table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
