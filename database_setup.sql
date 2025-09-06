-- Create users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    number VARCHAR(20) NOT NULL,
    password VARCHAR(255) NOT NULL,
    location VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create admins table
CREATE TABLE admins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default admin user (password: admin123)
INSERT INTO admins (email, password) VALUES 
('admin@hospital.com', 'pbkdf2:sha256:600000$admin123$hashed_password_here');

-- Create hospital tables
CREATE TABLE manipal_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    address TEXT NOT NULL,
    phone VARCHAR(20) NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    prediction VARCHAR(50),
    confidence DECIMAL(5,2),
    image_path VARCHAR(255),
    test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE apollo_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    address TEXT NOT NULL,
    phone VARCHAR(20) NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    prediction VARCHAR(50),
    confidence DECIMAL(5,2),
    image_path VARCHAR(255),
    test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE aster_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    address TEXT NOT NULL,
    phone VARCHAR(20) NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    prediction VARCHAR(50),
    confidence DECIMAL(5,2),
    image_path VARCHAR(255),
    test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); 