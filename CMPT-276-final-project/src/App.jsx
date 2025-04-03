import Header from "./Header"
import Body from "./Body"
import Contact from "./Contact"
import Footer from "./Footer"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import React from "react"

export default function App() {
  return(
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Body />} />
        <Route path="/contact" element={<Contact />} />
      </Routes>
      <Footer />
    </Router>
  )
}
