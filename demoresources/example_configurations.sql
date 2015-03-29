-- QSAR seed values
-- Purpose: These SQL statements populate the QSAR PostgreSQL database with supported
--configurations for use by the client.

-- populate the "Implementation" table
INSERT INTO "Implementation"("Impl_Desc") VALUES
('One-point crossover'),
('Two-point crossover'),
('Cut and Splice'),
('Uniform crossover'),
('Half uniform crossover'),
('Three parent crossover');

-- populate the "Evo_Alg" table
INSERT INTO "Evo_Alg"( "Evo_Alg_Name", "Evo_Alg_Desc") VALUES 
('GA', 'Genetic Alg'),
('PSO', 'Particle Swarm'),
('BPSO', 'Binary Particle Swarm'),
('DE-BPSO', 'Differential Evolution-Binary Particle Swarm');

-- populate the "Model" table
INSERT INTO "Model"("Model_Name", "Type_Name") VALUES
('MLR','linear'),
('SVM','linear'),
('PLSR','linear'),
('ANN','non-linear'),
('KNN','non-linear'),
('RF','non-linear'),
('TreeNet','non-linear'),
('MLR','linear'),
('SVM','linear');
