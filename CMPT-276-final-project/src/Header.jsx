import Logo from "./Factify-Logo.jpeg"
import { Link } from "react-router-dom"

export default function Header() {
    return (
     
      
        <div>
        <header className="header">
          <div className="container">
            <div className="logo">
            <Link to="/">
              <img src={Logo} alt="FACTIFY Logo" className="logo-image" />
            </Link>
              
            </div>
            <nav className="nav">
              <ul className="nav-list">
                <li><a href="#features">Features</a></li>
                <li><a href="#pricing">Pricing</a></li>
                <li><Link to="/contact">Contact</Link></li>
              </ul>
            </nav>
         
            <div className="actions">
              <button className="search-button">
                Search
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 21L16.65 16.65M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              <button className="sign-in-button">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M16 7C16 9.20914 14.2091 11 12 11C9.79086 11 8 9.20914 8 7C8 4.79086 9.79086 3 12 3C14.2091 3 16 4.79086 16 7Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 14C8.13401 14 5 17.134 5 21H19C19 17.134 15.866 14 12 14Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Sign In
              </button>
            </div>
          </div>
        </header>

        
        
      </div>
    )
  }