<h1>🌾 Monitoring Crop Health using Deep Learning and Grad-CAM++</h1>

<h2>📌 Project Objective:</h2>
<p>
This project simulates aerial crop health monitoring using deep learning and computer vision.  
The goal is to classify regions of crop images as <b>healthy</b> or <b>diseased</b> and visually highlight the affected areas using <b>Grad-CAM++ overlays</b>.
</p>

<h2>🧠 Models Used:</h2>
<ul>
  <li>✅ <b>EfficientNetB0 </b> – Lightweight CNN for plant disease detection</li>
  <li>✅ <b>EfficientNetB0 with CBAM</b> – Lightweight, attention-enhanced CNN optimized for better feature focus</li>
  <li>✅ <b>Grad-CAM++</b> – Visual explanation technique to highlight critical image regions contributing to the model's prediction</li>
</ul>

<h2>🗃️ Project Structure:</h2>

<pre>
MONITORING-CROP-HEALTH-USING-COMPUTER-VISION/
├── data/                 📂 Sample datasets and image folders
│   └── test_images/      📂 Test images for Grad-CAM++ inference
│   └── Output_Sample/     📂 GradCAM output samples
├── models/               📂 Saved model weights (.h5)
│   ├── efficientnet_b0_cbam_model.h5
│   └── efficientnet_b0_final_model.h5
├── notebooks/            📓 Training and inference notebooks
│   ├── EfficientNet_b0_training.ipynb
│   ├── Efficient_net_b0_CBAM_training.ipynb
│   ├── project-evaluation.ipynb
│   └── project-inference.ipynb
├── src/                  ⚙️ Source code files
│   ├── EfficientNet_b0_CBAM_Architecture.py
│   ├── EfficientNet_b0_Architecture.py
│   └── gradcam.py
│   └── Training_model.py
│   └── evaluation.py
│   └── inference.py
├── README.md             📄 Project documentation
</pre>

<h2>📦 Requirements:</h2>

<ul>
<li>Python (3.7+ recommended)</li>
<li>TensorFlow</li>
<li>OpenCV</li>
<li>NumPy</li>
<li>Matplotlib</li>
<li>Scikit-Learn</li>
</ul>

<h2>📥 Dataset Requirement:</h2>
<ul>
<li>📌 Download the <b>Plant Disease Dataset</b> from Kaggle:</li>
<li><a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset" target="_blank">New Plant Diseases Dataset (Augmented)</a></li>
<li>Extract the <code>train</code> and <code>val</code> folders into your <code>data/</code> directory.</li>
</ul>

<h2>📦 Installation:</h2>
<ol>
<li>Make sure you have <b>Python 3.8+</b> installed.</li>
<li>Clone the repository:</li>
<pre><code>git clone https://github.com/varssha22/MONITORING-CROP-HEALTH-USING-COMPUTER-VISION.git
cd MONITORING-CROP-HEALTH-USING-COMPUTER-VISION</code></pre>
<li>Install required libraries:</li>
<pre><code>pip install -r requirements.txt</code></pre>
</ol>

<h2>🚀 Usage:</h2>

<h3>🔧 1. Train a Model</h3>
<p>Train <b>EfficientNetB0</b> baseline:</p>
<pre><code>python src/train.py --model efficientnetb0 --epochs 30 --save_dir models</code></pre>

<p>Or train <b>EfficientNetB0 with CBAM</b> attention mechanism:</p>
<pre><code>python src/train.py --model efficientnetb0_cbam --epochs 30 --save_dir models</code></pre>

<hr>

<h3>📝 2. Generate Classification Report</h3>
<pre><code>python src/evaluation.py --model efficientnetb0_cbam --weights_path models/best_model.h5</code></pre>
<p>Output report saved as <code>classification_report.txt</code></p>

<hr>

<h3>🌾 3. Grad-CAM++ Visualization for Test Images</h3>
<pre><code>python src/inference_gradcam.py --model efficientnetb0_cbam --weights_path models/best_model.h5</code></pre>
<p>Visualizations saved under <code>gradcam_outputs/</code> folder.</p>

<h4>📊 EfficientNet B0 Model Architecture</h4>
<img src="data/EfficientNet-Architecture-diagram.png" width=500> <br>

<h4>🌾 MBConv Block</h4>
<img src="data/MBConv-block-with-Squeeze-and-Excitation.png" width=500> <br>

<h4>🌾 CBAM Module Illustration</h4>
<img src="data/The-Convolutional-Block-Attention-Module-CBAM-The-upper-side-is-the-channel-attention.png" width=500> <br>

<h2>🛠️ Model Training:</h2>
<ul>
<li>Models built using <b>Keras API</b> with <b>TensorFlow backend</b>.</li>
<li>Custom EfficientNetB0 with CBAM integrated for enhanced attention to diseased areas.</li>
<li>Training metrics like accuracy and loss visualized using matplotlib.</li>
</ul>

<h2>📄 Output:</h2>
<ul>
<li>Trained models saved under <code>/models/</code> in <b>.h5 format</b>.</li>
</ul>

<h2>🎯 Inference & Visualization:</h2>
<ul>
<li>Grad-CAM++ applied to test images to visually explain model predictions.</li>
<li>Red overlays highlight unhealthy/diseased regions of crops.</li>
<li>Comparison of original images, heatmaps, and outlined regions provided.</li>
</ul>

<h2>📂 Output Samples:</h2>

<img src="data/Output_Sample/gradcam_output_18.jpg"> <br>
<img src="data/Output_Sample/gradcam_output_13.jpg"> <br>
<img src="data/Output_Sample/gradcam_output_20.jpg"> <br>

<h2>📈 Results:</h2>

<table border="1" cellpadding="5">
<thead>
<tr>
<th>Model</th>
<th>Accuracy</th>
<th>Parameters</th>
<th>Grad-CAM++ Compatible</th>
</tr>
</thead>
<tbody>
<tr>
<td>EfficientNetB0</td>
<td>100.0%</td>
<td>~7 Million 🔻</td>
<td>✅ Yes (Lightweight)</td>
</tr>
<tr>
<td>EfficientNetB0 + CBAM</td>
<td>99.0%</td>
<td>~7 Million 🔻</td>
<td>✅ Yes (Lightweight & Attention-Enhanced)</td>
</tr>
</tbody>
</table>

<h2>🤝 Acknowledgements:</h2>
<ul>
<li><a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset">Kaggle – New Plant Diseases Dataset (Augmented)</a></li>
<li><a href="https://arxiv.org/pdf/1905.11946">EfficeintNet Paper</a></li>
<li><a href="https://arxiv.org/pdf/1807.06521v2">CBAM Paper</a></li>
<li>TensorFlow, OpenCV, and Keras Teams</li>
</ul>

<h2>📬 Contact:</h2>
<p>
For questions or collaborations, reach me via <a href="https://github.com/varssha22" target="_blank">GitHub Profile</a>.
</p>
