import React, { useState } from 'react';
import Select, { components } from 'react-select';
import axios from "axios";
import diamondIcon from '../others/diamond-color-icon.png';
import FileUpload from './FileUpload';


const clarityOptions = [
  { value: 0, label: 'I1' },
  { value: 3, label: 'SI2' },
  { value: 2, label: 'SI1' },
  { value: 5, label: 'VS2' },
  { value: 4, label: 'VS1' },
  { value: 7, label: 'VVS2' },
  { value: 6, label: 'VVS1' },
  { value: 1, label: 'IF' }
];

const colorOptions = [
  { value: 0, label: 'D' },
  { value: 1, label: 'E' },
  { value: 2, label: 'F' },
  { value: 3, label: 'G' },
  { value: 4, label: 'H' },
  { value: 5, label: 'I' },
  { value: 6, label: 'J' }
];

function Diamonds(){
  const [carat, setCarat] = useState('');
  const [x, setX] = useState('');
  const [y, setY] = useState('');
  const [z, setZ] = useState('');
  const [color, setColor] = useState('');
  const [clarity, setClarity] = useState('');
  const [diamondPrice, setDiamondPrice] = useState('');
  const [showFileUpload, setShowFileUpload] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();

    if (!carat || !x || !y || !z || !color || !clarity) {
      alert("Please fill all the fields.");
      return;
    }

    const diamondData = {
        carat: parseFloat(carat),
        x: parseFloat(x),
        y: parseFloat(y),
        z: parseFloat(z),
        color: color.value,
        clarity: clarity.value
      };
    console.log(diamondData);


    try{
        let result = await axios.post("http://127.0.0.1:5000/predict", { data: diamondData})
        console.log(result.data.price)
        setDiamondPrice(result.data.price)
    }
    catch(err){
        console.log(err)
    }
    //   .then(response => console.log(response.data))
    //   .catch(error => console.error(error));
  };

  if (showFileUpload){
    return <FileUpload />
  }

  return (
    <form className="form" onSubmit={handleSubmit}>
    <div className='title'>
        <img src={diamondIcon} alt='Diamond icon'/>
        <h1>Diamond Price Predictor</h1>
    </div>
      <div>
        <label>Carat:</label>
        <input
          type="number"
          value={carat}
          onChange={(e) => setCarat(e.target.value)}
        />
      </div>
      <div>
        <label>X:</label>
        <input
          type="number"
          value={x}
          onChange={(e) => setX(e.target.value)}
        />
      </div>
      <div>
        <label>Y:</label>
        <input
          type="number"
          value={y}
          onChange={(e) => setY(e.target.value)}
        />
      </div>
      <div>
        <label>Z:</label>
        <input
          type="number"
          value={z}
          onChange={(e) => setZ(e.target.value)}
        />
      </div>
      <div>
        <label>Color:</label>
        <Select
          options={colorOptions}
          value={color}
          onChange={setColor}
        />
      </div>
      <div>
        <label>Clarity:</label>
        <Select
          options={clarityOptions}
          value={clarity}
          onChange={setClarity}
        />
      </div>
      <button type="submit">Predict Price</button>
      <button onClick={() => setShowFileUpload(true)}>Upload file</button>
      {diamondPrice != "" && <p className='result'>The predicted price of the diamond is <span className='price'>{diamondPrice}$</span></p>}
    </form>
  );
};

export default Diamonds;
