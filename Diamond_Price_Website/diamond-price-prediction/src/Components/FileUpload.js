import React, { useState } from "react";
import "../others/file-upload.css"
import diamondIcon from '../others/diamond-color-icon.png';
import axios from "axios";
import Diamonds from "./Diamonds";


function FileUpload(){

    const [file, setFile] = useState(null);
    const [back, setBack] = useState(false);

    function handleFileChange(e){
        setFile(e.target.files[0]);
        console.log(e.target.files[0]);
    }
    async function handleSubmit(e){
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', file);

        try{
            const result = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
              });
            console.log(result.data);
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
                <button onClick={() => setBack(true)}>Back</button>
            </form>
        </div>
    )
}

export default FileUpload;