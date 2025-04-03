import React, { useState } from 'react';
import MagneticElement from './MagneticElement';

export default function ContactForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
  });
  const [status, setStatus] = useState({
    submitted: false,
    submitting: false,
    info: { error: false, msg: null },
  });

  // Replace this with your Web3Forms access key
  const apiKey = 'YOUR_WEB3FORMS_API_KEY';

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus({ submitted: false, submitting: true, info: { error: false, msg: null } });

    try {
      const response = await fetch('https://api.web3forms.com/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          access_key: apiKey,
          name: formData.name,
          email: formData.email,
          message: formData.message,
          subject: 'New contact form submission',
        }),
      });

      const data = await response.json();

      if (response.status === 200) {
        setStatus({
          submitted: true,
          submitting: false,
          info: { error: false, msg: 'Message sent successfully!' },
        });
        setFormData({ name: '', email: '', message: '' });
      } else {
        setStatus({
          submitted: false,
          submitting: false,
          info: { error: true, msg: data.message || 'Something went wrong. Please try again.' },
        });
      }
    } catch (error) {
      setStatus({
        submitted: false,
        submitting: false,
        info: { error: true, msg: 'An error occurred. Please try again later.' },
      });
    }
  };

  return (
    <section className="contact-page">
      <div className="container">
        <div className="contact-header">
          <h1 className="contact-title">
            Get in <span className="highlight">Touch!</span>
          </h1>
          <p className="contact-subtitle">
            Have a question or feedback? We'd love to hear from you :)
          </p>
        </div>

        <div className="contact-form-container">
          <form onSubmit={handleSubmit} className="contact-form">
            <div className="form-group">
              <label htmlFor="name">Name</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Your name"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Your email address"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="message">Message</label>
              <textarea
                id="message"
                name="message"
                value={formData.message}
                onChange={handleChange}
                placeholder="Your message"
                rows="5"
                required
              />
            </div>

            <div className="form-submit">
              <MagneticElement strength={40}>
                <button
                  type="submit"
                  className="submit-button"
                  disabled={status.submitting}
                >
                  {status.submitting ? 'Sending...' : 'Send Message'}
                </button>
              </MagneticElement>
            </div>

            {status.info.error && (
              <div className="error-message">
                {status.info.msg}
              </div>
            )}

            {status.submitted && (
              <div className="success-message">
                {status.info.msg}
              </div>
            )}
          </form>
        </div>
      </div>
    </section>
  );
} 