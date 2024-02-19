use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

struct Particle {
    id: i32,
    particle_id: String,
    active: bool,
    keep_active: bool,
    total_time: i32,
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

    println!("Opened File: {}", filenamestring);

    let buf = BufReader::new(file);
    buf.lines()
        .map(|l| l.expect("Could not parse line"))
        .collect()
}

fn main() {
    let file_dir_1 = [
        "liggghts_s_curl5_2024-02-07/", // "curl_0/", "curl_5/",
    ];
    let file_dir_2 = [
        "cpi_s_curl5_7/", //,
    ];
    rayon::scope(|s| {
        for i in 0..file_dir_1.len() {
            for j in 0..file_dir_2.len() {
                s.spawn(move |_| {
                    let mut particles: Vec<Particle> = Vec::new();

                    let mut id_count = 0;

                    let filename =
                        file_dir_1[i].to_string() + &file_dir_2[j].to_string() + "collision.txt";

                    let lines = lines_from_file(filename);

                    for line in lines {
                        let stringvalues = line.split_whitespace().collect::<Vec<_>>();

                        if stringvalues[0] == "file" {
                            for n in 0..particles.len() {
                                if !particles[n].keep_active {
                                    particles[n].active = false;
                                }
                            }
                        }

                        if stringvalues[0] == "file" {
                            for n in 0..particles.len() {
                                particles[n].keep_active = false;
                            }
                        } else {
                            let mut found_particle = false;
                            for n in 0..particles.len() {
                                if line == particles[n].particle_id {
                                    if particles[n].active {
                                        found_particle = true;
                                        particles[n].keep_active = true;
                                        particles[n].total_time += 1;
                                        break;
                                    }
                                }
                            }
                            if !found_particle {
                                let new_particle = Particle {
                                    id: id_count,
                                    particle_id: line.clone(),
                                    active: true,
                                    keep_active: true,
                                    total_time: 1,
                                };
                                id_count += 1;
                                particles.push(new_particle)
                            }
                        }
                    }

                    let mut a = Vec::<i32>::new();

                    for n in 0..particles.len() {
                        a.push(particles[n].total_time);
                    }

                    let savefile =
                        (file_dir_1[i].to_string() + file_dir_2[j].clone() + "array.txt");

                    let mut f =
                        BufWriter::new(File::create(savefile).expect("Unable to create file"));

                    for num in a {
                        write!(f, "{}, ", num);
                    }
                });
            }
        }
    });
}
