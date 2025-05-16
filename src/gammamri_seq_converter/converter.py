"""Converter between pypulseq sequence and simulator yaml file
"""
import pypulseq as pp

# ADDITIONAL DEFAULT PARAMS
adc_dead_time: float = 0
gamma: float = 42.576e6
grad_raster_time: float = 10e-6
grad_unit: str = "Hz/m"
max_grad: float = 0
max_slew: float = 0
rf_dead_time: float = 0
rf_raster_time: float = 1e-6
rf_ringdown_time: float = 0
rise_time: float = 0
slew_unit: str = "Hz/m/s"


def read_seq(seq_filename: str):
    print(f"Read .seq file: {seq_filename}")
    seq = pp.Sequence()  # create empty seq
    seq.read(seq_filename)
    print(f"Seq grad raster time = {seq.grad_raster_time} [s]")

    out_seq_filename = "out_" + seq_filename
    min_time = 1e-6
    system = pp.Opts(grad_raster_time=min_time)
    print(f"System grad raster time = {system.grad_raster_time} [s]")

    seq2 = pp.Sequence(system=system)
    print(seq2.grad_raster_time)
    print(f"Seq2 (with system) grad raster time = {seq2.grad_raster_time} [s]")

    print(f"Write to {out_seq_filename}")
    seq2.write(out_seq_filename)

    seq3 = pp.Sequence()
    seq3.read(out_seq_filename)
    print(seq3.grad_raster_time)
    print(
        f"Seq3 (read from seq2 with system file) grad raster time = {seq3.grad_raster_time} [s]"
    )


def seq_to_yaml():
    print("seq_to_yaml")
