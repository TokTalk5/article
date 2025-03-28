import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import get_window

# Параметры сигнала
fs = 15000  # Частота дискретизации
t = np.arange(0, 1.0, 1.0/fs)  # Временная шкала
n = len(t)

# Выбор условий
print("Введите условия записи: 1 - студийные, 2 - уличные, 3 - помещение")
condition = int(input())

def generate_signal(condition, n, fs):
    # Генерация основного сигнала

# Дипломная функция
    np.random.seed()
    frequencies = np.random.uniform(200, 400, 5)
    phases = np.random.uniform(0, 2*np.pi, 5)
    amplitudes = np.random.uniform(0.5, 1.5, 5)
    signal = np.sum([a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amplitudes, frequencies, phases)], axis=0)

    # Генерация шума
    noise_frequencies = np.random.uniform(0, fs / 2, 10)
    noise_phases = np.random.uniform(0, 2*np.pi, 10)
    noise_amplitudes = np.random.uniform(0, 5, 10)
    noise = np.sum([a * np.sin(2 * np.pi * f * t + p) for f, p, a in zip(noise_frequencies, noise_phases, noise_amplitudes)], axis=0)
# Дипломная функция


    if condition == 1:
        # Студия
        noise_level = 0.01
        noise *= noise_level
    elif condition == 2:
        # Улица
        noise_level = 0.5
        noise *= noise_level
        traffic_noise = np.sin(2 * np.pi * 50 * t)
        noise += 0.3 * traffic_noise
    elif condition == 3:
        # Помещение
        noise_level = 0.1
        noise *= noise_level
        reverb_signal = np.zeros_like(signal)
        for i in range(1, 4):
            reverb_signal += 0.5**i * np.roll(signal, int(fs * 0.05 * i))  # Затухающие копии сигнала
        noise += reverb_signal

# Дипломная функция
    # Сложный сигнал с шумом
    complex_signal = signal + noise

    # Применение классического окна Хэмминга
    classic_window = get_window('hamming', n)
    windowed_signal_classic = complex_signal * classic_window
    spectrum_classic = np.fft.fft(windowed_signal_classic)
# Дипломная функция

    return windowed_signal_classic, spectrum_classic

# Загрузка реального аудиофайла
real_audio_path = "voice.wav"  # Укажите путь к вашему аудиофайлу
real_data, real_samplerate = librosa.load(real_audio_path, sr=None)

# Применение окна Хэмминга к реальному аудио
real_window = get_window('hamming', len(real_data))
windowed_real_signal = real_data * real_window
spectrum_real = np.fft.fft(windowed_real_signal)

# Генерация сигнала в зависимости от условий
windowed_signal_classic, spectrum_classic = generate_signal(condition, n, fs)

# Визуализация спектров
plt.figure(figsize=(14, 10))

# Спектр сгенерированного сигнала
plt.subplot(2, 1, 1)
frequencies_synth = np.fft.fftfreq(n, 1/fs)
plt.plot(frequencies_synth[:n//2], np.abs(spectrum_classic[:n//2]))
plt.title('Спектр сгенерированного сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim(0, 8000)
plt.ylim(0, np.max(np.abs(spectrum_classic[:n//2])))

# Спектр реального голосового сигнала
plt.subplot(2, 1, 2)
frequencies_real = np.fft.fftfreq(len(real_data), 1/real_samplerate)
plt.plot(frequencies_real[:len(real_data)//2], np.abs(spectrum_real[:len(real_data)//2]))
plt.title('Спектр реального голосового сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim(0, 8000)
plt.ylim(0, np.max(np.abs(spectrum_classic[:n//2])))  

plt.tight_layout(pad=3.0)
plt.show()
