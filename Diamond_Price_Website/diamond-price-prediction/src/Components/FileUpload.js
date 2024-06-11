import React, { useState } from "react";
import "../others/file-upload.css"
import diamondIcon from '../others/diamond-color-icon.png';
import axios from "axios";
import Diamonds from "./Diamonds";


function FileUpload(){

    const [file, setFile] = useState(null);
    const [back, setBack] = useState(false);
    const [url, setUrl] = useState();
    const [downloadAttr, setDownloadAttr] = useState();

    function handleFileChange(e){
        setFile(e.target.files[0]);
        console.log(e.target.files[0]);
    }
    async function handleSubmit(e){
        e.preventDefault();

        if (!file) {
            alert("Please select a file to upload.");
            return;
          }
        
        const formData = new FormData();
        formData.append('file', file);

        try{
            const result = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
                responseType:"blob",
              });
            console.log(result.data);
            console.log(url);

            setUrl(window.URL.createObjectURL(new Blob([result.data], { type: 'text/csv' })));
            setDownloadAttr('predicted_prices.csv');
            // const link = document.createElement('a');
            // link.href = url;
            // link.setAttribute('download', 'predicted_prices.csv');
            // document.body.appendChild(link);
            // link.click();
            // document.body.removeChild(link);

            
        }
        catch(err){
            console.log(err);
        }
    }

    if (back){
        return <Diamonds />
    }

    return(
        <div className="file-upload-container">
            <form onSubmit={handleSubmit}>
                <div className='title'>
                    <img src={diamondIcon} alt='Diamond icon'/>
                    <h1>Diamond Price Predictor</h1>
                </div>
                <div>
                    <label>Upload your file here</label>
                    <input
                    type="file"
                    onChange={handleFileChange}
                    />
                </div>
                <button type="submit">Upload</button>
                <button onClick={() => setBack(true)}>Back</button><br/>
                {url != null && <a className="download"
                href={url}
                download={downloadAttr}>Download Price Predicted File</a>}
            </form>
        </div>
    )
}

export default FileUpload;