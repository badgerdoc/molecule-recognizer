from pathlib import Path

import click

from image_captioning.cli.commands import create_new_model, resume_from
from image_captioning.utils.prediction import get_prediction


@click.group()
def cli():
    pass


@click.option('--pipeline', '-p', help='Pipeline config file', required=True, type=Path)
@click.option('--encoder', '-e', help='Encoder config file', required=True, type=Path)
@click.option('--decoder', '-d', help='Decoder config file', required=True, type=Path)
@click.command()
def new(pipeline, encoder, decoder):
    """Creates new model from pipeline, encoder, decoder cfg files (yml) and starts train loop"""
    click.echo(f'Creating a new model:\n{pipeline}\n{encoder}\n{decoder}\n')
    create_new_model(pipeline, encoder, decoder)


@click.option('--checkpoint', '-c', help='Checkpoint path to resume from', required=True, type=Path)
@click.option(
    '--skip_steps', '-s',
    help='To resume from approximately the same step, where process was interrupted, default=True',
    type=bool,
    default=True
)
@click.command()
def resume(checkpoint, skip_steps):
    """Resumes the training from checkpoint, skips already trained steps (by default, can be disabled)"""
    click.echo(f'Resuming from checkpoint {checkpoint}')
    resume_from(checkpoint, skip_steps)


@click.option('--checkpoint', '-c', help='Checkpoint path', required=True, type=Path)
@click.command()
def evaluate(checkpoint):
    # TODO: evaluation pipeline is not implemented yet
    click.echo(f'Evaluation for checkpoint: {checkpoint}')


@click.option('--img', help='Image path', required=True, type=Path)
@click.option('--checkpoint', '-c', help='Checkpoint path', required=True, type=Path)
@click.command()
def inference(img, checkpoint):
    """Image captioning inference on a single image"""
    click.echo(f'\nInference for image: {img}')
    get_prediction(img, checkpoint)


cli.add_command(new)
cli.add_command(resume)
cli.add_command(evaluate)
cli.add_command(inference)


if __name__ == "__main__":
    cli()
