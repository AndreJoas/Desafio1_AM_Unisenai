# Pipeline FFT + Classificação de Sinais

Este projeto implementa um pipeline de processamento de sinais utilizando FFT (Fast Fourier Transform), normalização, PCA e modelos de classificação com **Optuna** para otimização de hiperparâmetros.

O projeto foi adaptado para rodar tanto em ambiente local quanto no **Google Colab**, incluindo visualizações em 2D e 3D com **Plotly** e gráficos de acurácia/confusion matrix.

---

## Estrutura do Repositório

```
project-root/
│
├─ dados/ # Arquivos numpy (.npy) com os sinais e labels
│ ├─ x_1500_10.npy
│ ├─ y_1500_10.npy
│ ├─ z_1500_10.npy
│ └─ gt_1500_10.npy
│
├─ historico/ # Histórico de execuções (JSON)
│
├─ static/images/ # Imagens e plots gerados
│
├─ app.py # Script principal Flask
├─ requirements.txt # Dependências Python
└─ README.md```
