from predictor.audio_model import AudioModel

class AudioSegmenter(AudioModel):
    def _format_output(self, y):
        y = self.encoder.inverse_transform(y)
        list_detections = list()
        last_value = y[0]
        last_from = 0
        # Unificacion de las predicciones por clase y creacion de las detecciones
        step = 1
        for i, prediction in enumerate(y[1:], start=1):
            if prediction != last_value:  # and prob[i][classes.index(Y_pred[i])] > 0.65:
                list_detections.append({"value": last_value, "from": last_from * step, "to": i * step})
                last_from = i
                last_value = prediction
        list_detections.append({"value": last_value, "from": last_from * step, "to": len(y) * step})
        return list_detections

if __name__ == '__main__':
    import sys
    print(AudioSegmenter(sys.argv[2]).predict(sys.argv[1]))