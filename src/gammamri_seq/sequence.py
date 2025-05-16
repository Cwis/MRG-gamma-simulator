import math

from gammamri_seq.pulse_seq import PulseSeq


class Sequence:
    def __init__(
        self,
        title: str = "",
        pulse_seq=[],
        repetition_time: float = 0,
        number_repetitions: int = 0,
        precision: int = 5,
        time_unit_multiplier: float = 1e3,
    ):
        self.title: str = title
        self.pulse_seq = []
        self.repetition_time = repetition_time
        self.number_repetitions = number_repetitions

        self._precision = precision
        self._time_unit_multiplier = time_unit_multiplier

        self.keys = ["title", "pulseSeq", "TR", "nTR"]

    def add_pulse_seq(self, pulse_seq: PulseSeq):
        self.pulse_seq.append(pulse_seq)

    def dict(self):
        # list_pulse_seq = [ps.dict() for ps in self.pulse_seq]
        return dict(
            zip(
                self.keys,
                [
                    self.title,
                    [
                        ps.dict(self._precision, self._time_unit_multiplier)
                        for ps in self.pulse_seq
                    ],
                    round(
                        math.ceil(self.repetition_time * self._time_unit_multiplier),
                        self._precision,
                    ),
                    self.number_repetitions,
                ],
            )
        )
