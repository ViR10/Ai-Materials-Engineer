class Material:
    def __init__(self, name, density, hardness, uts):
        self.name = name
        self.density = density
        self.hardness = hardness
        self.uts = uts  # Ultimate Tensile Strength (MPa)

    def show_info(self):
        return f"{self.name} | Density: {self.density} g/cm3 | Hardness: {self.hardness} HB | UTS: {self.uts} MPa"


class MaterialDatabase:
    def __init__(self, filename="materials_data.txt"):
        self.materials = []          # List jo saare Material objects rakhegi
        self.filename = filename     # File ka naam
        self.load_from_file()        # Program start hote hi data load karo

    def add_material(self, name, density, hardness, uts):
        new_material = Material(name, density, hardness, uts)
        self.materials.append(new_material)
        self.save_to_file()
        print(f"'{name}' database mein add ho gaya.")

    def delete_material(self, name):
        for m in self.materials:
            if m.name.lower() == name.lower():
                self.materials.remove(m)
                self.save_to_file()
                print(f"'{name}' delete ho gaya.")
                return
        print(f"'{name}' database mein mila hi nahi.")

    def search_material(self, name):
        for m in self.materials:
            if m.name.lower() == name.lower():
                print("Mil gaya:", m.show_info())
                return
        print(f"'{name}' database mein nahi hai.")

    def show_all(self):
        if len(self.materials) == 0:
            print("Database khali hai.")
        else:
            for m in self.materials:
                print(m.show_info())

    def save_to_file(self):
        try:
            with open(self.filename, "w") as file:
                for m in self.materials:
                    file.write(f"{m.name},{m.density},{m.hardness},{m.uts}\n")
        except Exception as e:
            print("File save karte waqt error aaya:", e)

    def load_from_file(self):
        try:
            with open(self.filename, "r") as file:
                for line in file:
                    parts = line.strip().split(",")

                    # Invalid line skip karo
                    if len(parts) != 4:
                        continue

                    name = parts[0]
                    density = float(parts[1])
                    hardness = float(parts[2])
                    uts = float(parts[3])

                    self.materials.append(
                        Material(name, density, hardness, uts)
                    )

        except FileNotFoundError:
            print("Koi purani file nahi mili, naya database start ho raha hai.")


def main():
    db = MaterialDatabase()

    while True:
        print("\n--- Material Database ---")
        print("1. Add Material")
        print("2. Delete Material")
        print("3. Search Material")
        print("4. Show All Materials")
        print("5. Exit")

        choice = input("Apna choice number likho: ")

        if choice == "1":
            name = input("Material ka naam: ")
            density = float(input("Density (g/cm3): "))
            hardness = float(input("Hardness (HB): "))
            uts = float(input("UTS (MPa): "))

            db.add_material(name, density, hardness, uts)

        elif choice == "2":
            name = input("Kis material ko delete karna hai: ")
            db.delete_material(name)

        elif choice == "3":
            name = input("Kis material ko search karna hai: ")
            db.search_material(name)

        elif choice == "4":
            db.show_all()

        elif choice == "5":
            print("Program band ho raha hai.")
            break

        else:
            print("Galat choice, dobara try karo.")


# Program yahan se start hota hai
if __name__ == "__main__":
    main()