import React from 'react';
import logo from './logo.svg';
import './App.css';
import axios from 'axios';

class App extends React.Component {
    state = {
        image: null,
        name: ""
    }
    submitFn = (e) => {
        e.preventDefault();
        const data = new FormData();
        data.append('image', this.state.image);
        data.append('name', this.state.name);
        axios.post("api/", data, { // receive two parameter endpoint url ,form data 
        })
        .then(res => { // then print response status
            console.log(res.statusText)
        })
    }
    render() {
        return (
            <div className="App">
                <form onSubmit={this.submitFn} encType="multipart/form-data">
                    <input type="text" name="name" value={this.state.name} onChange={(e) => {this.setState({"name":e.target.value})}} />
                    <input type="file" name="image" onChange={(e) => {this.setState({image:e.target.files[0]})}} />
                    <input type="submit" />
                </form>
                <br />
                <br />
                <br />
                <Recognize />
            </div>
        );
    }
}

class Recognize extends React.Component {
    state = {
        image: null,
        name: "",
        status: false,
        result: []
    }
    submitFn = (e) => {
        e.preventDefault();
        this.setState({status:true});
        const data = new FormData();
            data.append('image', this.state.image);
            axios.post("api/recog/", data, { // receive two parameter endpoint url ,form data 
        })
        .then((resp) => {
            console.log(resp);
            if(resp.data.error==false)
                this.setState({result: resp.data.data,status:false});
        }).catch((err) => {
            this.setState({status:false});
        });
    }
    render() {
        return (
            <form onSubmit={this.submitFn} encType="multipart/form-data">
                <input type="file" name="image" onChange={(e) => {this.setState({image:e.target.files[0]})}} />
                <input type="submit" />
                {
                    this.state.status?
                        <p>Loading</p>:null
                }
                <ul>
                    {
                        this.state.result.map((item) => 
                            <li>{item}</li>
                        )
                    }
                </ul>
            </form>
        );
    }
}

export default App;
