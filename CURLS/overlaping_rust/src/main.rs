use nalgebra::{Matrix2, Vector2, Vector3, Vector4};
use rayon::prelude::*;
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    process,
};
use xml::reader::{EventReader, XmlEvent};
const RADIUS: f64 = 0.139183e-3;

#[derive(Debug)]
struct Particle {
    quat: Vector4<f64>,
    points: Vec<Vector3<f64>>,
    id: u32,
    max: Vector3<f64>,
    min: Vector3<f64>,
    size: Vector3<f64>,
}

impl Particle {
    fn clone(&self) -> Particle {
        let particle = Particle {
            quat: self.quat.clone(),
            points: self.points.clone(),
            id: self.id,
            max: self.max,
            min: self.min,
            size: self.size,
        };
        return particle;
    }

    fn set_aabb(&mut self) {
        if self.points.is_empty() {
            return;
        }

        self.max = self.points[0] + Vector3::new(RADIUS, RADIUS, RADIUS);
        self.min = self.points[0] - Vector3::new(RADIUS, RADIUS, RADIUS);

        for point in self.points.iter() {
            self.max[0] = self.max[0].max(point[0] + RADIUS);
            self.max[1] = self.max[1].max(point[1] + RADIUS);
            self.max[2] = self.max[2].max(point[2] + RADIUS);
            self.min[0] = self.min[0].min(point[0] - RADIUS);
            self.min[1] = self.min[1].min(point[1] - RADIUS);
            self.min[2] = self.min[2].min(point[2] - RADIUS);
        }

        self.size = self.max - self.min;
    }

    fn rotation_self(&mut self) {
        for p in self.points.iter_mut() {
            *p = point_rotation_by_quaternion(p.clone(), self.quat);
        }
    }

    fn translate_self(&mut self, translation: Vector3<f64>) {
        for p in self.points.iter_mut() {
            let new_p = *p + translation;
            *p = new_p;
        }
    }
}

fn point_rotation_by_quaternion(pos: Vector3<f64>, quat: Vector4<f64>) -> Vector3<f64> {
    let r = [0.0, pos[0], pos[1], pos[2]];
    let q = [quat[0], quat[1], quat[2], quat[3]];
    let q_conj = [quat[0], -1.0 * quat[1], -1.0 * quat[2], -1.0 * quat[3]];
    let temp = quaternion_mult(quaternion_mult(q, r), q_conj);
    return Vector3::new(temp[1], temp[2], temp[3]);
}

fn quaternion_mult(q: [f64; 4], r: [f64; 4]) -> [f64; 4] {
    return [
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
        r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
        r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0],
    ];
}
fn check_collision(particle_1: &Particle, particle_2: &Particle) -> Vec<f64> {
    let mut overlaps = Vec::new();

    for point1 in particle_1.points.iter() {
        for point2 in particle_2.points.iter() {
            let p1 = point1;
            let p2 = point2;
            let dif = p2 - p1;
            let distance = dif.magnitude();
            if distance < RADIUS * 2.0 {
                let overlap = (RADIUS * 2.0 - distance);
                overlaps.push(overlap);
            }
        }
    }

    return overlaps;
}

fn lines_from_file(filename: String) -> Vec<String> {
    let filenamestring: String = filename.clone();
    let file = match File::open(filename) {
        Ok(file) => file,
        Err(err) => {
            println!("Error: {}", err);
            std::process::exit(1);
        }
    };

    // println!("Opened File: {}", filenamestring);

    let buf = BufReader::new(file);
    buf.lines()
        .map(|l| l.expect("Could not parse line"))
        .collect()
}

fn get_particles_from_file(
    quatfilename: String,
    temp_pos: Vec<Vector3<f64>>,
    radius: f64,
) -> Vec<Particle> {
    let quatf = lines_from_file(quatfilename);

    let mut particles = Vec::new();

    let mut size = Vec::new();
    for i in 0..temp_pos.len() {
        size.push(radius);
    }

    for line in quatf.into_iter() {
        let results: Vec<&str> = line.split_whitespace().collect();

        let mut particle: Particle = Particle {
            quat: Vector4::new(
                results[5].parse::<f64>().unwrap(),
                results[6].parse::<f64>().unwrap(),
                results[7].parse::<f64>().unwrap(),
                results[8].parse::<f64>().unwrap(),
            ),
            points: temp_pos.clone(),
            id: results[1].parse::<u32>().unwrap(),
            max: Vector3::zeros(),
            min: Vector3::zeros(),
            size: Vector3::zeros(),
        };

        particle.rotation_self();
        particle.translate_self(Vector3::new(
            results[2].parse::<f64>().unwrap(),
            results[3].parse::<f64>().unwrap(),
            results[4].parse::<f64>().unwrap(),
        ));
        particle.set_aabb();

        particles.push(particle);
    }

    return particles;
}

fn main() {
    let args: Vec<String> = env::args().collect();

    println!("Interlocking Detection code written by Elizabeth Suehr");

    if args.len() < 11 {
        println!("Please include arguments: ./interlocking_detection domain particle curl1 simulation liggghts_s_curl1 volumefractions cpi_s_curl1_0 startcount 100000");
        process::exit(1);
    }

    let mut domain_collections = false;
    let mut particle_collection = false;
    let mut simulation_collection = false;
    let mut volumefration_collection = false;
    let mut startcount_collection = false;

    let mut curl_directory = vec![];
    let mut vf_directory = vec![];
    let mut start_count = vec![];
    let mut particle_files = vec![];
    let mut domain_size = vec![];

    for a in args.iter() {
        if a == "domain" {
            domain_collections = true;
        } else if a == "particle" {
            particle_collection = true;
            domain_collections = false;
        } else if a == "simulation" {
            simulation_collection = true;
            particle_collection = false;
        } else if a == "volumefractions" {
            volumefration_collection = true;
            simulation_collection = false;
        } else if a == "startcount" {
            volumefration_collection = false;
            startcount_collection = true;
        } else {
            if domain_collections {
                println!("domain {}", a);
                domain_size.push(a.parse::<f64>().unwrap())
            }
            if particle_collection {
                println!("particle_files {}", a);
                particle_files.push(a)
            }

            if simulation_collection {
                println!("curl folders {}", a);
                curl_directory.push(a)
            }

            if volumefration_collection {
                println!("vf folders {}", a);
                vf_directory.push(a)
            }

            if startcount_collection {
                println!("start count {}", a);
                let num: u32 = a.parse().unwrap();
                start_count.push(num)
            }
        }
    }

    rayon::scope(|s| {
        for (curl, pfile) in curl_directory.iter().zip(particle_files.iter()) {
            for (vf_i, (vf, start)) in vf_directory.iter().zip(start_count.iter()).enumerate() {
                s.spawn(move |_| {
                    let pf = lines_from_file("./".to_owned() + pfile);

                    let mut temp_particle: Vec<Vector3<f64>> = Vec::new();
                    let mut radius = 0.0;
                    for line in pf {
                        let results: Vec<&str> = line.split_whitespace().collect();
                        temp_particle.push(Vector3::new(
                            results[0].parse::<f64>().unwrap(),
                            results[1].parse::<f64>().unwrap(),
                            results[2].parse::<f64>().unwrap(),
                        ));
                        radius = results[3].parse::<f64>().unwrap(); //WARNING, this assumes all the same size radius.
                    }

                    let mut particles: Vec<Particle> = Vec::new();

                    let domain = Vector3::new(0.008636, 0.008636, 0.004318);

                    let write_file_name = "./".to_string()
                        + &curl.to_string()
                        + "/"
                        + &vf.to_string()
                        + "/"
                        + "overlap.txt";

                    let mut f = File::create(write_file_name).expect("Unable to create file");

                    let mut file_output = "".to_string();

                    for i in 0..100 {
                        file_output += &("file ".to_string() + &i.to_string() + "\n");
                        let file_name = "./".to_string()
                            + &curl.to_string()
                            + "/"
                            + &vf.to_string()
                            + "/cpi_"
                            + &(start + i * 5000).to_string()
                            + ".txt";
                        particles.clear();
                        particles =
                            get_particles_from_file(file_name, temp_particle.clone(), radius);

                        let mut edge_particles = Vec::new();
                        let mut k = 0;
                        while k < particles.len() {
                            if particles[k].max[0] > domain[0]
                                || particles[k].max[2] > domain[2]
                                || particles[k].min[0] < 0.0
                                || particles[k].min[2] < 0.0
                            {
                                if particles[k].max[0] > domain[0] {
                                    if particles[k].max[2] > domain[2] {
                                        let mut particle1 = particles[k].clone();

                                        particle1.translate_self(Vector3::new(-domain.x, 0.0, 0.0));
                                        let mut particle2 = particles[k].clone();
                                        let mut particle3 = particle1.clone();

                                        particle2.translate_self(Vector3::new(0.0, 0.0, -domain.z));
                                        particle3.translate_self(Vector3::new(0.0, 0.0, -domain.z));

                                        edge_particles.push(particle1);
                                        edge_particles.push(particle2);
                                        edge_particles.push(particle3);
                                    } else if particles[k].min[2] < 0.0 {
                                        let mut particle1 = particles[k].clone();

                                        particle1.translate_self(Vector3::new(-domain.x, 0.0, 0.0));
                                        let mut particle2 = particles[k].clone();
                                        let mut particle3 = particle1.clone();

                                        particle2.translate_self(Vector3::new(0.0, 0.0, domain.z));
                                        particle3.translate_self(Vector3::new(0.0, 0.0, domain.z));

                                        edge_particles.push(particle1);
                                        edge_particles.push(particle2);
                                        edge_particles.push(particle3);
                                    } else {
                                        let mut particle1 = particles[k].clone();
                                        particle1.translate_self(Vector3::new(-domain.x, 0.0, 0.0));
                                        edge_particles.push(particle1);
                                    }
                                } else if particles[k].min[0] < 0.0 {
                                    if particles[k].max[2] > domain[2] {
                                        let mut particle1 = particles[k].clone();

                                        particle1.translate_self(Vector3::new(domain.x, 0.0, 0.0));
                                        let mut particle2 = particles[k].clone();
                                        let mut particle3 = particle1.clone();

                                        particle2.translate_self(Vector3::new(0.0, 0.0, -domain.z));
                                        particle3.translate_self(Vector3::new(0.0, 0.0, -domain.z));

                                        edge_particles.push(particle1);
                                        edge_particles.push(particle2);
                                        edge_particles.push(particle3);
                                    } else if particles[k].min[2] < 0.0 {
                                        let mut particle1 = particles[k].clone();

                                        particle1.translate_self(Vector3::new(domain.x, 0.0, 0.0));
                                        let mut particle2 = particles[k].clone();
                                        let mut particle3 = particle1.clone();

                                        particle2.translate_self(Vector3::new(0.0, 0.0, domain.z));
                                        particle3.translate_self(Vector3::new(0.0, 0.0, domain.z));

                                        edge_particles.push(particle1);
                                        edge_particles.push(particle2);
                                        edge_particles.push(particle3);
                                    } else {
                                        let mut particle1 = particles[k].clone();
                                        particle1.translate_self(Vector3::new(domain.x, 0.0, 0.0));
                                        edge_particles.push(particle1);
                                    }
                                } else if particles[k].max[2] > domain[2] {
                                    let mut particle1 = particles[k].clone();
                                    particle1.translate_self(Vector3::new(0.0, 0.0, -domain.z));
                                    edge_particles.push(particle1);
                                } else if particles[k].min[2] < 0.0 {
                                    let mut particle1 = particles[k].clone();
                                    particle1.translate_self(Vector3::new(0.0, 0.0, domain.z));
                                    edge_particles.push(particle1);
                                }

                                edge_particles.push(particles.remove(k));
                            } else {
                                k += 1;
                            }
                        }

                        particles.retain(|particle| {
                            if particle.max[0] > domain[0]
                                || particle.max[1] > domain[1]
                                || particle.max[2] > domain[2]
                                || particle.min[0] < 0.0
                                || particle.min[1] < 0.0
                                || particle.min[2] < 0.0
                            {
                                return false;
                            } else {
                                return true;
                            }
                        });

                        // particles are already "Cleaned" because we started from body positions
                        let mut cleaned_edge_particles = edge_particles;

                        particles.sort_by(|p1, p2| (p1.max[1]).partial_cmp(&p2.max[1]).unwrap());
                        cleaned_edge_particles
                            .sort_by(|p1, p2| (p1.max[1]).partial_cmp(&p2.max[1]).unwrap());

                        let mut overlaps = Vec::new();
                        // Check between inner particles
                        for j in 0..particles.len() {
                            for k in j + 1..particles.len() {
                                if particles[j].max[1] < particles[k].min[1] {
                                    break;
                                }

                                let mut overlaps_to_add =
                                    check_collision(&particles[j], &particles[k]);
                                overlaps.append(&mut overlaps_to_add);
                            }
                        }
                        // // Check inner particles with outer particles
                        // for j in 0..particles.len() {
                        //     for k in 0..cleaned_edge_particles.len() {
                        //         if particles[j].max[1] < cleaned_edge_particles[k].min[1] {
                        //             break;
                        //         }

                        //         let mut overlaps_to_add =
                        //             check_collision(&particles[j], &particles[k]);
                        //         overlaps.append(&mut overlaps_to_add);
                        //     }
                        // }
                        println!("{:?}", overlaps)
                        // f.write_all(file_output.as_bytes())
                        //     .expect("Unable to write data");
                        // file_output.clear();
                    }
                });
            }
        }
    })
}
