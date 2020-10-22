class ExperimentData():

    def __init__(self, eid):
        2 + 2

    # def get_injection_hemisphere():

    def flip(self):
        """Reflects experiment along midline.

        Returns
        -------
        self - flipped experiment
        """

        self.injection_qmasked = self.injection_qmasked[..., ::-1]
        self.projection_qmasked = self.projection_qmasked[..., ::-1]
        self.injection_signal = self.injection_signal[..., ::-1]
        self.projection_signal = self.projection_signal[..., ::-1]
        self.injection_signal_true = self.injection_signal_true[..., ::-1]
        # self.projection_signal_true = self.projection_signal_true[..., ::-1]
        self.injection_fraction = self.injection_fraction[..., ::-1]
        self.data_quality_mask = self.data_quality_mask[..., ::-1]

        # return self
