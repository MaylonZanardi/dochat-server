
<body>
	<h1>doChat</h1>
	<p>This repository contains the source code of a web application that allows users to upload files, which are then processed by a Python script.</p>  
  <h2>Installation</h2>
<p>To install this application, follow the steps below:</p>
<ol>
	<li>Clone this repository:</li>
	<pre>git clone https://github.com/yourusername/ZTDocsGPT.git</pre>
	<li>Navigate to the root directory of the project:</li>
	<pre>cd ZTDocsGPT</pre>
	<li>Install the required Python packages:</li>
	<pre>pip install -r requirements.txt</pre>
</ol>

<h2>Usage</h2>
<p>To run the application, execute the following command:</p>
<pre>python app.py</pre>

<p>The application can be accessed by opening a web browser and navigating to <code>http://localhost:5000</code>.</p>

<h2>Functionality</h2>
<p>The web application provides a form where the user can select one or more files to upload. Once the user has selected the files and submitted the form, the files are saved to the <code>scripts/inputs</code> directory and the Python script <code>scripts/ingest.py</code> is executed with the <code>--dir</code> argument set to the <code>scripts/inputs</code> directory. The output of the script is displayed to the user.</p>
