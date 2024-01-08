import pandas as pd
import numpy as np
import os
import glob
import sys
import argparse
from scipy import stats

"""
from sklearn import preprocessing
from sklearn import model_selection as ms

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit

from sklearn.svm import SVC
from sklearn import tree

from sklearn.metrics import balanced_accuracy_score as bas

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
"""


# Questions when this has worked:   Feature selection + more metrics 


def main():
    # Command line argument setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dt", "--decision_trees", help="use decision trees classifier")
    argParser.add_argument("-svm", "--support_vector_machine", help="use support vector machine classifier")
    args = argParser.parse_args()

    dataloader = Dataloader(sys.argv[2])

    if args.decision_trees is not None:
        decision_trees = Decision_Trees(dataloader)
        #decision_trees.train()
        

    if args.support_vector_machine is not None:
        svm = SVM(dataloader)
        #svm.train()
        


    print("Finished")



class Decision_Trees():

    """Decision Tree classification"""
    
    def __init__(self, dataloader):
        self.train_set, self.target = dataloader.get()

    def train(self):
        print("Classifier: Decision Trees")

        # TODO: implement decision tree classifier (https://scikit-learn.org/stable/modules/tree.html)

        #X = self.train_set
        #y = self.target

        X = np.random.randn(20,3)                                          #Dummy
        y = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])            #Dummy

        X = preprocessing.normalize(X, norm="l2")
        X_train, X_val, y_train, y_val = ms.train_test_split(X, y, train_size=0.8, random_state=42)

        clf = tree.DecisionTreeClassifier(max_depth=6, max_leaf_nodes=13)
        clf.fit(X_train, y_train)

        cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        cross_val_score = ms.cross_val_score(clf, X_train, y_train, cv=cv)

        predict = clf.predict(X_val)

        print()
        print("cross validation scores: ",cross_val_score)
        print("accuracy score: ",bas(y_val,predict))




class SVM():

    """Support Vector Machine Classification"""

    def __init__(self, dataloader):
        self.train_set, self.target = dataloader.get()

    def train(self):
        print("Classifier: Support Vector Machine")

        #(https://scikit-learn.org/stable/modules/svm.html)

        #X = self.train_set
        #y = self.target

        X = np.random.randn(20,3)                                          #Dummy
        y = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])            #Dummy

        X = preprocessing.normalize(X, norm="l2")
        X_train, X_val, y_train, y_val = ms.train_test_split(X, y, train_size=0.8, random_state=42)

        svc = SVC()

        rand_list = {"clf__kernel":["rbf"],
                     "clf__C": stats.uniform(1, 100),
                     "clf__gamma": stats.uniform(0.01, 1)}
        
        
        grid_list = {"clf__kernel":["rbf"],
                     "clf__C": np.arange(2, 10, 2),
                     "clf__gamma": np.arange(0.1, 1, 0.2)}

        model = Pipeline([
            ("sampling", RandomOverSampler(random_state=0)),
            #("sampling", RandomUnderSampler(random_state=0)),
            #("sampling", SMOTE()),
            ("clf", svc)
            ])
        
        cv = ms.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        # Random search
        clf = RandomizedSearchCV(model, param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = cv, random_state = 42, scoring = "balanced_accuracy", verbose=49) 
        clf.fit(X_train, y_train) 
        clf.cv_results_
        
        """
        # Gridsearch
        clf = GridSearchCV(model, param_grid = grid_list, n_jobs = 4, cv = cv, scoring = "balanced_accuracy", verbose=49) 
        clf.fit(X_train, y_train) 
        clf.cv_results_
        """

        predict = clf.predict(X_val)

        print("accuracy score: ",bas(y_val,predict))




class Dataloader():

    """Load and prepare Dataloader for sklearn classifiers"""


    def __init__(self,path):
        """Get all train data csv files from folder"""

        self.path = os.path.join(os.getcwd(), str(path))


    def get(self):
        """return train data csvs concatenated as one numpy array and targets"""

        train_set = None
        target = None

        train_data_dict = {}

        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, str(filename))
            file = pd.read_csv(file_path, low_memory=False)
            train_data_dict[filename] = file

        recipes = train_data_dict["recipes.csv"]
        requests = train_data_dict["requests.csv"]
        reviews = train_data_dict["reviews.csv"]
        diet = train_data_dict["diet.csv"]

        train_set = requests.merge(reviews, on=["AuthorId","RecipeId"])
        train_set = train_set[train_set["Like"].notna()]                    # Filter samples with no Like value (not usable for training)
        train_set["Like"] = train_set["Like"].astype(int)                   # Convert Like to False=0 and True=1
        train_set = train_set.merge(diet,on="AuthorId")
        train_set = train_set.merge(recipes,on="RecipeId")

        #train_set.to_csv("train_set_raw.csv",",")

        # /--------------------------------


        # TODO Niclas

        
        omnivor_list = ["chicken broth", "Copycat Taco Bell Seasoned Beef", "chicken", "lean ground beef", "ground beef", "bacon bits", "roasting chickens", "chicken breast fillets", "shrimp", "parma ham", "boneless chicken breast", "bacon", "boneless skinless chicken breast", "chicken bouillon granules", "thick slab bacon", "pork sausage", "skinless chicken thighs", "lean bacon", "sausages", "prosciutto", "chicken breasts", "chorizo sausage", "ham", "lamb chops", "Thai fish sauce", "cooked pork", "turkey", "Worcestershire sauce", "light chunk tuna in water", "boneless skinless chicken breast halves", "oyster sauce", "tuna", "chicken bouillon cubes", "salmon fillet", "low sodium chicken broth", "sausage", "whole chicken", "low sodium beef broth", "reduced-sodium chicken broth", "boneless chicken breasts", "hamburger", "smoked trout fillets", "sea scallops", "raw shrimp", "mussels", "cooked ham", "crawfish", "albacore tuna", "turkey carcass", "turkey breast", "skinless chicken breasts", "hot Italian sausage", "swordfish steaks", "boneless skinless chicken breasts", "anchovy fillets", "ground pork", "fish sauce", "frying chicken", "strawberry Jell-O gelatin dessert", "character(", "bulk Italian sausage", "beef eye round", "beef round steak", "beef bouillon cube", "bacon fat", "sugar-free raspberry gelatin", "tuna in olive oil", "beef tenderloin", "miniature marshmallows", "deli corned beef", "condensed beef broth", "top round steaks", "lamb", "ground turkey", "medium shrimp", "prosciutto di Parma", "beef brisket", "boneless skinless chicken", "low-sodium instant chicken bouillon granules", "ground beef round", "chicken wings", "crawfish tails", "boneless skinless chicken breast half", "skinless chicken pieces", "boneless pork loin", "smoked trout", "boneless chicken breast halves", "lump crabmeat", "salmon fillets", "smoked sausage", "lemon gelatin", "chicken thigh", "pink salmon", "poultry seasoning", "Jello gelatin", "barbecued pork", "mini marshmallows", "sweet Italian sausage links", "salmon", "halibut steaks", "tuna in water", "chicken pieces", "beef liver", "chicken liver", "chicken fat", "mild Italian sausage", "boneless leg of lamb", "fat free chicken broth", "squid", "sirloin tip roast", "beef bouillon cubes", "turkey breast tenderloins", "miniature marshmallow", "rump roast", "instant chicken bouillon granules", "anchovies", "smoked streaky bacon", "gelatin", "corned beef", "boneless lamb shoulder", "catfish fillets", "sugar-free orange gelatin", "boneless pork roast", "boneless pork chops", "large shrimp", "Canadian bacon", "bacon drippings", "chicken drumsticks", "white meat chicken", "red snapper fillets", "beef chuck", "hot sausage", "ground lamb", "beef stew meat", "clams", "duck fat", "chicken breast halves", "beef tenderloin steaks", "bouillon cubes", "boneless skinless chicken thighs", "bulk pork sausage", "chicken breast", "breakfast sausage", "salt cod fish", "smoked bacon", "country ham", "lean ground turkey", "fat-free chicken broth", "corned beef brisket", "skinless chicken breast halves", "chicken bouillon", "boneless pork loin roast", "lamb cutlets", "condensed new england clam chowder", "tri-tip roast", "duck breasts", "turkey broth", "turkey sausage", "cooked corned beef", "Italian sausage", "chicken bouillon granule", "skinless chicken breast", "boneless beef round steak", "beef bouillon granules", "beef roast", "beef", "minced beef", "beef top round steak", "nonfat beef broth", "boneless pork shoulder", "fresh ahi tuna", "lamb breast", "stewing lamb", "boneless pork chop", "lean beef chuck", "beef suet", "fresh shrimp", "frozen shrimp", "beef flank steak", "tasso", "lamb shoulder", "roast beef", "skinless chicken", "andouille sausage", "serrano ham", "ham bone", "Southeast Asian fish sauce", "rich chicken broth", "bulk sausage", "white wine worcestershire sauce", "trassi oedang", "crawfish meat", "lamb fillets", "whole chickens", "chicken bouillon cube", "beef steaks", "boneless salmon fillets", "smoked salmon", "trout fillets", "beef sirloin", "italian sweet sausage", "boneless beef roast", "stewing beef", "venison steak", "shoulder lamb chops", "minced clams", "bacon grease", "turkey kielbasa", "chunk light tuna", "chicken bouillon powder", "tuna steaks", "mild sausage", "hamburger meat", "worcestershire sauce for chicken", "cod fish fillet", "sugar-free strawberry gelatin", "anchovy paste", "tiger shrimp", "chicken meat", "marshmallows", "ground chicken", "chicken tenderloins", "cod fish fillets", "lobsters", "streaky bacon", "lamb stew meat", "ham steak", "chicken thighs", "strawberry gelatin", "crisp bacon", "fresh salmon", "pork", "salmon steaks", "boneless beef top sirloin steak", "Swanson chicken broth", "boneless beef cube", "chicken piece", "reduced-sodium ham", "crawfish tail", "ham hock", "beef bones with marrow", "low-sodium beef bouillon cubes", "short rib of beef", "sweet Italian sausage link", "hot Italian sausage link", "lamb racks", "sugar-free lemon gelatin", "lamb leg steaks", "chicken breast halve", "jumbo shrimp", "roasting chicken", "oven-roasted deli chicken", "Polish sausage", "baby clams", "salt pork", "top round beef", "country-style pork ribs", "flounder fillets", "swordfish steak", "rabbit", "skinless salmon fillet", "halibut fillets", "fresh tuna", "racks of lamb", "gelatin powder", "chicken fillets", "boneless chicken", "canned salmon", "fresh pork loin roast", "chicken livers", "boneless pork ribs", "deli roast beef", "rack of lamb", "ham hocks", "chicken cutlets", "large marshmallows", "smoked ham hock", "red snapper", "smoked ham", "boneless pork loin steaks", "sole fillets", "chicken parts", "ducklings", "smoked link sausage", "instant beef bouillon", "lamb blade chops", "lamb stock", "chicken portions", "london broil beef", "rump steak", "bottom round beef roast", "mahi mahi fillets", "canned tuna", "mahi mahi", "sirloin beef", "chicken legs-thighs", "ground sausage", "whiting fish fillets", "instant chicken bouillon", "deli ham", "ham steaks", "chicken legs", "marshmallow miniatures", "lean ground lamb", "beef bouillon powder", "duck", "swordfish fillets", "mortadella", "lobster meat", "tuna salad", "canned chicken broth", "bouillon", "leg of lamb", "herring fillets", "sausage meat", "chicken flavor instant bouillon", "lean hamburger", "ham slices", "lamb shanks", "liver", "lean lamb fillets", "eye of round roast", "small shrimp", "homemade chicken broth", "chunk tuna", "manila clams", "lime Jell-O gelatin", "smoked ham hocks", "boneless beef sirloin", "turkey meat", "beef schnitzel", "half and half milk", "skinless chicken breast half", "littleneck clams", "clam", "boneless beef chuck roast", "stewing chicken", "chicken gizzard", "piri-piri", "sea bass fillets", "smoked back bacon", "cooked lamb", "haddock fillets", "chicken necks", "chicken back", "orange gelatin", "chicken giblets", "fresh lump crabmeat", "red salmon", "polska kielbasa", "quail", "Chile Verde Con Cerdo (Green Chili With Pork", "cod", "New England clam chowder", "Polish kielbasa", "Spanish ham", "prosciutto ham", "sirloin tip steaks", "lobster tails", "lean salt pork", "lamb chop", "solid white tuna", "Versatile Roast Beef in the Crock Pot", "sugar-free peach gelatin mix", "tuna in brine", "chicken fillet", "back bacon", "condensed chicken broth", "all beef wieners", "fresh sea scallops", "canned anchovy fillets", "tuna steak", "boneless beef rump roast", "lean ham", "halibut fillet", "turkey slices", "beef ribs", "lox", "extra-large shrimp", "filet of beef", "hot chicken broth", "boneless pork", "live lobsters", "boneless chicken breast half", "beef shank", "top round roast", "chicken thigh fillets", "cherry gelatin", "steamer clams", "center-cut pork chops", "trout", "boneless pork loin chops", "bay shrimp", "boneless round steak", "fresh crabmeat", "turkey tenderloins", "suet", "shrimp paste", "sirloin tip steak", "beef mince", "catfish fillet", "sockeye salmon", "salmon steak", "speck", "lobster", "anchovy fillet", "fillets of sole", "broiler-fryer chickens", "sea bass fillet", "braunschweiger sausage", "veal liver", "anchovy", "chicken backs", "lean lamb", "boneless beef cubes", "instant bouillon granules", "albacore tuna in water", "beef steak", "Chorizo", "chicken thigh pieces", "lamb steaks", "lamb shank", "beefsteak tomato", "salmon salad", "fryer chickens", "Italian pork sausage", "raspberry Jell-O gelatin", "lobster tail", "peking duck", "solid white tuna packed in water", "small marshmallows", "top round steak", "brown and serve sausages", "boneless pork sirloin", "nam pla", "ducks", "perch", "Simple and Healthy Poached Salmon", "sardine fillets", "raspberry gelatin powder", "Shrimp Stock", "rainbow trout", "tuna fillets", "nonfat chicken broth", "frying chickens", "white turkey meat", "pork sausage link", "boneless pork blade roast", "fresh sea scallop", "unsalted side pork", "low-sodium chicken bouillon cubes", "hot Italian sausage links", "lobster head", "cherrystone clams", "black cherry gelatin", "ahi tuna steaks", "clams in shell", "lardons", "barbecued chicken", "beef rib", "chicken gumbo soup", "lean rump steak", "Basic White Stock", "u- 12 shrimp", "tuna fish", "lean beef", "aspic", "lean lamb steaks", "beef tenderloin steak", "steelhead trout", "boneless lamb roast", "lamb loin chops", "sirloin lamb chops", "broiler-fryer chicken", "lean beef chuck roast", "regular hamburger", "chicken cutlet", "boneless salmon fillet", "crayfish", "pork jowl", "black-eyed peas with bacon", "boneless center cut pork chops", "beef fat", "lamb rib chops", "boneless pork cutlets", "country sausage", "tripe", "chicken thigh fillet", "beef sirloin steak", "unsmoked bacon", "grape gelatin", "boneless beef top round steak", "bresaola", "boneless beef top loin steaks", "clam broth", "canned clams", "skinless boneless pheasant breast halves", "colored miniature marshmallows", "shin beef", "lamb loin chop", "beef sirloin steaks", "pheasant breast", "sole fillet", "borscht", "tuna packed in oil", "lamb necks", "broiler chicken", "jumbo lump crab meat", "round tip roast", "Starkist tuna", "chicken breast fillet", "alligator tail steaks", "kosher gelatin", "beef short ribs with bones", "round tip steak", "tuna in vegetable oil", "fresh swordfish steaks", "butterflied leg of lamb", "crayfish tails", "Taco Filling (Ground Beef", "Beef Machaca", "Tex-Mex Carne Asada", "Carnitas (Authentic", "Shredded Chicken for Enchiladas", "Fiesta Lengua (Tongue", "turkey slice", "boneless duck breast", "boneless beef top round", "large unpeeled shrimp", "chicken leg", "condensed chicken gumbo soup", "lean boneless lamb", "prosciutto rind", "nuoc nam", "pheasant", "boned lamb", "marshmallow peeps", "boneless center cut pork loin roast", "mahi mahi fillet", "spiral cut ham", "boneless beef top sirloin steaks", "sugar-free cherry gelatin", "boneless ham", "boneless beef short ribs", "duck legs", "fatty bacon", "sage sausage", "crawfish tail meat", "garlic sausage", "ready-to-serve beef broth", "turkey breast tenderloin", "fillet of sole", "red snapper fillet", "boneless pork top loin", "skinless chicken piece", "boneless lamb", "yellowfin tuna steak", "boneless atlantic salmon fillet", "partridge breasts", "Chinese barbecue pork", "berry blue gelatin mix", "dark chicken meat", "boneless duck breast halves", "Morningstar Farms Meal Starters chicken strips", "wild ducks", "frozen crab", "goose", "soft-shell clams", "Knorr chicken bouillon", "small clams", "Bacon Pastry Crust", "rindless smoked streaky bacon", "trout fillet", "beef round tip steaks", "low-fat turkey kielbasa", "beef flavored Rice-A-Roni", "roll of pork sausage", "chicken breast half", "orange-pineapple flavored gelatin", "stew beef chunk", "white crab meat", "beef drippings", "alligator meat", "chestnut meats", "boneless beef brisket", "beef silverside", "eel", "boneless beef chuck shoulder pot roast", "extra lean beef", "Bahia-Mar Resort's Mangolade Duck Sauce", "lean lamb stew meat", "pork sausage links", "broiler chickens", "Mini Bacon Meatballs", "beef kielbasa", "canned baby clams", "tri-tip steak", "canned broth", "canned shrimp", "no-salt-added chicken broth", "boneless pork cutlet", "Starkist lemon and cracked pepper tuna fillets", "frozen crabmeat", "boneless bottom round roast", "peach gelatin", "raw chicken", "yellowfin tuna steaks", "chicken feet", "reduced-fat kielbasa", "yellowfin tuna fillet", "pork broth", "jamon serrano", "roasted ancho chile", "calf liver", "boneless pork loin chop", "apricot gelatin", "fruit flavored gelatin", "lamb steak", "boneless center cut pork chop", "chicken drumstick", "commercial low-sodium chicken broth", "quahogs", "boneless lean pork", "young roasting chickens", "boneless skinless cod", "cod steaks", "low-sodium beef bouillon cube", "loin lamb", "boneless boston pork roast", "lean beef round", "frozen lobster tails", "ham shank", "chicken neck", "lamb liver", "lamb rack", "fish bouillon cube", "boneless lamb chops", "lean round steak", "joint of beef", "Starkist sweet and spicy tuna", "small marshmallow", "shrimp bouillon cube", "salted salmon", "boneless duck breasts", "Agave Glazed Bacon", "fat pork", "lamb leg chops", "lamb gravy", "low joule gelatin", "boneless skinned chicken breast", "wild strawberry gelatin", "gelatin sheets", "boneless lamb loin", "boneless beef top round steaks", "marshmallow bits", "salted herrings", "Lemon-Grilled Chicken Breasts", "red gelatin", "blueberry gelatin", "salmon tails", "Chinese barbecued duck", "boneless skinned chicken breasts", "pickled herring", "hot breakfast sausage patty", "chicken gizzards", "low-fat ham", "duck carcass", "baby chicken", "duck giblets", "Redondo Iglesias jamsn serrano", "Ham Stock (Pressure Cooker", "fresh pork hocks", "lamb ribs", "spring chicken", "fresh crabs", "sugar-free lime gelatin", "strawberry-banana gelatin", "cooked duck", "blachan", "clam chowder", "Peppered Pork Loin", "Grilled Greek Chicken Breasts", "full cut round steaks", "black cod steaks", "pork bouillon cube", "boneless sirloin tip roast", "roast turkey meat", "lamb rib", "ham fat", "rabbit joints", "Campbell's chicken gumbo soup", "blackberry gelatin", "fish bouillon cubes", "Basic Chicken Stock", "northern pike fillets", "turkey stuffing", "97% fat-free cooked ham", "fryer chicken", "squid ring", "lamb backstraps", "herring fillet", "low-sodium chicken bouillon cube", "andouille chicken sausage", "turkey steaks", "dried morels", "sugar-free black cherry gelatin", "low-fat kielbasa", "yellowfin tuna fillets", "Nuoc Cham (Vietnamese Spicy Fish Sauce", "abalone", "snoek", "low-sodium ham", "lamb stock cube", "low-salt ham", "turkey fat", "dried shrimp paste", "cottage roll", "ham shanks", "fatty pork", "beef tip roast", "small clam", "veal broth", "Morningstar Farms Better" 
        , "Burgers", "bottom round steaks"]

        vegetarisch_list = ["butter", "cheddar cheese", "Velveeta cheese", "sour cream", "unsalted butter", "milk", "sharp cheddar cheese", "eggs", "sweetened condensed milk", "heavy whipping cream", "dark Creme de Cacao", "Amarula cream liqueur", "chocolate ice cream", "cheese", "heavy cream", "low-fat yogurt", "light mayonnaise", "egg", "cream cheese", "Cool Whip", "swiss cheese", "camembert cheese", "parmesan cheese", "honey", "phyllo pastry", "fat-free cottage cheese", "light butter", "2% buttermilk", "white chocolate chips", "reduced-fat alfredo sauce", "cottage cheese", "ricotta cheese", "chicken-flavored vegetarian seasoning", "vegetarian refried beans", "mozzarella cheese", "provolone cheese", "fat-free mayonnaise", "creme fraiche", "evaporated milk", "barbecue sauce", "hard-boiled egg", "mayonnaise", "hard-boiled eggs", "gruyere cheese", "reduced-fat sour cream", "nonfat milk", "plain low-fat yogurt", "buttermilk", "Miracle Whip", "plain yogurt", "monterey jack cheese", "Cotija cheese", "feta cheese", "reduced-fat mozzarella cheese", "salted butter", "fat free sour cream", "part-skim mozzarella cheese", "real butter", "monterey jack and cheddar cheese blend", "cream of tartar", "pistachio ice cream", "vanilla ice cream", "light cream cheese", "romano cheese", "low-fat buttermilk", "yogurt", "marshmallow creme", "American cheese", "white chocolate", "ghee", "instant vanilla pudding", "low fat cottage cheese", "bittersweet chocolate", "chocolate chips", "low-fat cheddar cheese", "French vanilla pudding mix", "low-fat milk", "parmigiano-reggiano cheese", "fat free cream cheese", "1% low-fat milk", "low-fat plain yogurt", "cream sherry", "reduced-fat vanilla ice cream", "gorgonzola", "focaccia bread", "jalapeno jack cheese", "dark chocolate", "bocconcini", "small curd cottage cheese", "feta", "skim milk", "Fontina cheese", "part-skim ricotta cheese", "light sour cream", "chocolate", "Miracle Whip light", "monterey jack pepper cheese", "half-and-half cream", "vanilla instant pudding mix", "instant chocolate pudding mix", "blue cheese", "low-fat mayonnaise", "instant pistachio pudding mix", "Stilton cheese", "plain nonfat yogurt", "Cool Whip Lite", "ice cream", "Bisquick", "chocolate fudge pudding mix", "yoghurt", "mascarpone cheese", "sharp Canadian cheddar cheese", "nonfat plain yogurt", "sweet butter", "farmer cheese", "chocolate flavor instant pudding and pie filling mix", "ice cream sandwiches", "Limburger cheese", "Roquefort cheese", "marshmallow cream", "dry milk", "chive & onion cream cheese", "low-fat sharp cheddar cheese", "fat-free buttermilk", "Hellmann's mayonnaise", "full-fat milk", "frozen shredded hash browns", "asiago cheese", "fat-free cool whip", "Velveeta reduced fat cheese product", "fat-free cheddar cheese", "fat-free ricotta cheese", "nonfat sour cream", "non-dairy whipped topping", "buttermilk baking mix", "halloumi cheese", "colby-monterey jack cheese", "2% low-fat milk", "mozzarella string cheese", "Velveeta Mexican cheese", "sharp American cheese", "queso fresco", "low-fat parmesan cheese", "fresh mozzarella cheese", "powdered milk", "low-fat sour cream", "canned milk", "mixed cheese", "apple butter", "sourdough English muffins", "sharp white cheddar cheese", "custard", "low-fat swiss cheese", "reduced-fat cheddar cheese", "Cheese Whiz", "parmigiano", "fontina", "reduced-fat swiss cheese", "plain fat-free yogurt", "fresh parmesan cheese", "clear honey", "fat free vanilla ice cream", "bittersweet chocolate piece", "instant pudding mix", "hash browns", "Kittencal's Marinara Pasta Sauce (Vegetarian", "irish cream", "mild cheddar cheese", "English muffins", "extra-sharp cheddar cheese", "reduced-fat cream cheese", "extra-large eggs", "strawberry ice cream", "extra-large egg", "chicken stove top stuffing mix", "sugar-free instant vanilla pudding mix", "sugar-free vanilla pudding mix", "sugar-free instant banana pudding mix", "non-fat powdered milk", "Cool Whip Free", "liquid honey", "chocolate pudding", "lowfat mozzarella cheese", "chocolate instant pudding", "light ricotta cheese", "alfredo sauce", "vanilla yogurt", "nonfat yogurt", "havarti cheese", "smoked cheddar cheese", "strawberry cream cheese", "nonfat cheddar cheese", "Velveeta shells and cheese dinner", "smoked gouda cheese", "low-fat Quark cheese", "lemon yogurt", "raw egg yolks", "fat-free parmesan cheese", "fat free mozzarella cheese", "cream cheese spread", "2% evaporated milk", "low-fat ricotta cheese", "fat-free evaporated milk", "non fat small curd cottage cheese", "lite evaporated milk", "2% cheddar cheese", "mascarpone", "Nestle sweetened condensed milk", "Carnation Evaporated Milk", "ganache", "colby", "firm butter", "whole buttermilk", "white cheddar cheese", "chocolate piece", "skim milk powder", "muenster cheese", "reduced-fat sharp cheddar cheese", "buffalo mozzarella", "smoked mozzarella cheese", "processed cheese", "white sauce", "non-fat vanilla yogurt", "haddock fillet", "chocolate curls", "low-fat cream cheese", "cream-style cottage cheese", "homogenized milk", "Baileys Irish Cream", "Kittencal's No-Fail Buttery Flaky Pie Pastry/Crust", "lemon pudding mix", "mozzarella cheddar blend cheese", "fat-free sugar-free instant vanilla pudding mix", "pudding", "coffee ice cream", "kasseri cheese", "bouillon cube", "white chocolate baking bar", "asadero cheese", "instant lemon pudding", "lowfat parmesan cheese", "cheese spread", "jumbo eggs", "red caviar", "black caviar", "baker's unsweetened chocolate squares", "light cheddar cheese", "amaretti cookie", "bechamel sauce", "creamed honey", "low fat mozzarella", "English muffin", "fat-free American cheese", "baking chocolate", "instant milk", "manchego cheese", "2% milk", "cheese slice", "colby cheese", "cheesecake flavor instant pudding and pie filling mix", "reduced-sodium Worcestershire sauce", "whole milk ricotta cheese", "gouda cheese", "bittersweet chocolate", "mint chocolate chip ice cream", "1% low-fat chocolate milk", "sugar-free white chocolate pudding mix", "white Creme de Cacao", "potato & cheese pierogies", "alfredo sauce mix", "pignolis", "instant coconut cream pudding mix", "reduced-fat milk", "2% fat cottage cheese", "large curd cottage cheese", "sugar-free vanilla ice cream", "light monterey jack cheese", "banana ice cream", "vegetarian chicken pieces", "1% fat cottage cheese", "aged cheddar cheese", "cheese slices", "sugar-free instant chocolate pudding mix", "fat-free Miracle Whip", "light boursin cheese", "chocolate graham cracker crumbs", "instant butterscotch pudding mix", "premade puff pastry", "whipped butter", "low-fat vanilla yogurt", "instant banana pudding mix", "ladyfinger", "taleggio", "vegetarian fat free sugar-free instant vanilla pudding mix", "sweet milk", "instant nonfat dry milk powder", "instant lemon pudding mix", "chocolate milk", "reduced-fat jarlsberg cheese", "process American cheese", "low-fat alfredo sauce", "medium cheddar", "bagels", "lowfat swiss cheese", "red leicester cheese", "sugar-free instant pudding mix", "pre-shredded mozzarella cheese", "edam cheese", "prepared honey-dijon barbecue sauce", "fat-free vanilla pudding", "no-carb cheddar cheese", "raw honey", "fat free sharp cheddar cheese", "date honey", "vegetarian hot dogs", "light mozzarella cheese", "raspberry ice cream", "lactose-free milk", "chocolate bar", "vegetarian gelatin", "hash brown", "Canadian cheddar cheese", "sweet unsalted butter", "Make Your Own Greek Yoghurt", "cheddar cheese powder", "vegetarian bacon", "fat free sugar-free instant cheesecake pudding mix", "pierogi", "chocolate pudding mix", "vegetarian hamburger patties", "Yogurt Cheese - 'Labanee'", "prepared pudding", "dry buttermilk", "reduced-fat extra sharp cheddar cheese", "Jello Instant Vanilla Pudding Mix", "southern style hash browns", "amaretti cookies", "Nido milk", "chocolate bars", "vegetarian sausage patties", "vegetarian oyster sauce", "sour cream substitute", "brick cheese", "Homemade Lavender Honey", "toffee ice cream", "pistachio pudding mix", "boursin cheese", "milk solids", "vegetarian ground beef", "reduced-fat baking mix", "butterscotch pudding mix", "vegetarian worcestershire sauce", "French vanilla ice cream", "eggnog ice cream", "orange blossom honey", "dry parmesan cheese", "chihuahua cheese", "skim evaporated milk", "sour cream and chive flavored cream cheese", "Jell-O pudding mix", "hard cheese", "low-fat low-sodium swiss cheese", "low fat  sweetened condensed milk", "low-moisture part-skim mozzarella cheese", "chocolate-covered graham cracker cookies", "heavy sweet cream", "Simple Honey Mustard Salad Dressing", "thyme honey", "Quark", "instant mashed potatoes with sour cream and chives", "cream cheese with chives", "smoked gruyere cheese", "remoulade sauce", "dry curd cottage cheese", "Martha White buttermilk", "white chocolate curls", "cheshire cheese", "dark cooking chocolate", "Homemade Rich Fresh RICOTTA Cheese", "2% mozzarella cheese", "boudoir biscuits", "instant white chocolate pudding and pie filling mix", "ice cream sandwich", "buckwheat honey", "Mexican vegetarian ground meat substitute", "German chocolate bar", "white chocolate pudding mix", "lemon flavor instant pudding and pie filling", "aged white cheddar cheese", "reduced-fat ricotta cheese", "3 Legume Butter", "prepared chocolate pudding", "cinnamon ice cream", "fresh ricotta", "German chocolate", "fat-free sugar-free white chocolate pudding mix", "fat-free sugar-free vanilla pudding mix", "American cheese spread", "medium sharp cheddar", "Rich Homemade Ranch Dressing", "reduced fat romano cheese", "low-fat monterey jack pepper cheese", "instant chocolate fudge pudding", "Kamut&reg; Cookies for Creme", "parmesan-romano cheese mix", "vegetarian chili", "filo pastry", "caerphilly cheese", "lancashire cheese", "Jell-O Oreo instant pudding & pie filling mix", "low-fat ricotta", "Crescent Roll Dough (Bread Machine", "dry non-fat buttermilk", "vegetarian parmesan cheese", "half & half light cream", "frozen yogurt", "whole milk mozzarella", "light cheese", "bleu cheese spread", "raclette cheese", "butterscotch pudding", "vegan mozzarella cheese", "roquefort blue cheese", "Hellmanns Mayonnaise", "fat free sugar-free instant chocolate fudge pudding mix", "cream cheese with green onion", "chocolate graham cracker", "bagel", "Best Foods Mayonnaise", "mozzarella-provolone cheese blend", "low-fat vanilla ice cream", "reduced-fat honey graham crackers", "sweet creamy butter", "dark semi-sweet chocolate", "white chocolate bark", "Jell-O chocolate fudge flavor pudding and pie filling", "skim milk ricotta cheese", "white chocolate baking squares", "sugar-free instant pistachio pudding", "cheesecake flavor instant pudding and pie filling", "vegetarian beef substitute", "fat-free cream cheese", "fresh mozzarella ball", "imported white chocolate", "baby swiss cheese", "fresh mozzarella balls", "light vanilla ice cream", "low-fat ice cream", "low-fat chocolate ice cream", "danish blue cheese", "Kraft processed cheese slices", "vegetarian instant vanilla pudding mix", "instant dry milk powder", "Ww Vegetable With Alfredo Sauce", "low-fat American cheese", "bite-size fresh mozzarella cheese balls", "peach yogurt", "parmesan asiago and romano cheese blend", "banana cream pudding and pie filling mix", "low-fat monterey jack cheese", "spaetzle noodles", "low-fat biscuit mix", "nonfat cottage cheese", "part-skim cottage cheese", "5% fat ricotta cheese", "chocolate Cool Whip", "low-fat creme fraiche", "large-curd cottage cheese", "sugar-free instant butterscotch pudding mix", "mild cheese", "half cream", "parmesan cheese", "montasio cheese", "Kraft 100% Parmesan Cheese", "cultured buttermilk", "Milnot Condensed Milk", "mocha ice cream", "butter pecan ice cream", "fat free cheese", "sugar-free fat-free butterscotch pudding", "Chocolate Crumb Crust", "pickled eggs", "double Gloucester cheese", "processed swiss cheese", "peppermint ice cream", "cheddar cheese cubes", "chocolate chip ice cream", "pudding mix", "vegetarian baked beans", "mozzarella cheese cubes", "praline ice cream", "processed cheese spread", "queso anejo", "fat-free vegetarian refried beans", "Greek feta cheese", "heather honey", "fat-free sugar-free instant chocolate pudding mix", "low-fat small-curd cottage cheese", "prune butter", "1% fat buttermilk", "vegetarian bologna", "mixed cheeses", "hash brown patties", "light processed cheese", "nonfat dry milk solid", "Cajun Portobello Sandwich With Avocado and Remoulade", "full-cream milk", "frozen miniature phyllo tart shells", "jumbo egg", "jalapeno havarti cheese", "cantal cheese", "quorn sausage", "saga blue cheese", "comte cheese", "swiss provolone cheese mix", "sugar free instant coconut pudding mix", "low-carb milk", "Sweetened Condensed Milk Substitute for Diabetics", "vegetarian meatballs", "german chocolate cake mix with pudding", "vegetarian Stilton cheese", "vegetarian bouillon powder", "mexican processed cheese sauce", "light jarlsberg cheese", "whole wheat English muffin", "masa dough", "Gluten Free Buttermilk Biscuits", "Copycat Cinnabon Rolls With Icing", "amaretti cookie crumbs", "vegetarian marshmallows", "garlic and cheese flavored croutons", "robiola cheese", "20% sour cream", "wensleydale cheese", "prepared sugar-free vanilla pudding", "Basic French Tart Dough/Pate Brisee (Dorie Greenspan", "cookies & cream ice cream", "vegetarian pudding", "Snail Butter", "sugar-free chocolate pudding mix", "whole grain English muffin", "chocolate fudge instant pudding mix", "bel paese cheese", "tahini sesame butter", "Morningstar Farms vegetarian buffalo wings", "masago smelt roe", "parmesan and mozzarella pasta sauce", "potato & cheese pierogi", "cabrales cheese", "vegetarian chicken flavored broth mix", "vegetarian beef broth mix", "jumbo egg yolks", "processed cheese food", "marshmallow whip", "vegetarian mayonnaise", "Anejo cheese", "peach ice cream", "pineapple-coconut ice cream", "french-style ladyfinger cookies", "fat-free monterey jack cheese", "pouring custard", "miniature bagels", "Focaccia", "Honey Muffins", "Baker#s Special Dry Milk", "queso sauce", "Herb  Butter", "vegetarian chicken substitute", "sugar-free instant chocolate fudge pudding mix", "coconut cream pudding mix", "light irish cream liqueur", "sugar-free instant lemon pudding", "sugar free pistachio pudding mix", "The Ultimate Creamy Blue Cheese Dressing &amp; Dip", "paneer cheese", "vegetarian beef strips", "burata cheese", "Blender Hollandaise Sauce", "reduced-fat mild cheddar cheese", "instant devil#s food pudding mix", "mozzarella-cheddar blend cheese", "2% large-curd cottage cheese", "prepared vanilla pudding", "vegetarian sausage links", "sugar-free fat-free banana cream pudding mix", "vegetarian blue cheese", "nonfat parmesan cheese", "Greek Tzatziki", "fat-free sugar-free instant white chocolate pudding mix", "light cream cheese with chives and onions", "instant butter pecan pudding mix", "tortellini cheese pasta", "instant mint-chocolate pudding mix", "Chez Panisse Almond Cake", "Lenotre Pastry Cream", "Kittencal's Easy Creamy White Glaze", "vegetarian chicken broth", "Kittencal's Perfect Pesto", "Tart Dough (sweet", "Cilantro Cream", "Easy Buttercream Icing", "Sargento. ChefStyle Cheddar Cheese", "No - Berry Strawberry Rhubarb Cake", "sugar free fat free French vanilla pudding and pie filling mix", "sage derby cheese", "instant devil's food pudding", "fat-free swiss cheese", "vegetarian salami", "instant mashed potatoes with butter and herbs", "sourdough English muffin", "vegetarian chicken soup mix", "Crumb Cake Mix", "creamed shortening", "Appenzeller cheese", "Miracle Whip from Mayonnaise", "mimolette cheese", "Lavender-Rose Honey", "vegetarian vegetable soup", "Classic Buttercream Frosting", "cheddar cheese cube", "Philadelphia Cream Cheese", "pecorino romano cheese", "Blue Cheese Vinaigrette", "maytag blue cheese", "reduced-fat monterey jack cheese"]
        
        train_set = train_set[train_set['RecipeIngredientParts'] != 'character(0)'] 
        train_set['RecipeIngredientParts'] = train_set['RecipeIngredientParts'].apply(lambda x: x[2:-1] if x[0] == 'c' else x)
        train_set['RecipeIngredientParts'] = train_set['RecipeIngredientParts'].str.replace('\\','',)
        train_set['RecipeIngredientParts'] = train_set['RecipeIngredientParts'].str.replace('"','',)
        train_set['RecipeIngredientParts'] = train_set['RecipeIngredientParts'].str.split(',')

        train_set = train_set[train_set['RecipeIngredientQuantities'] != 'character(0)'] 
        train_set['RecipeIngredientQuantities'] = train_set['RecipeIngredientQuantities'].apply(lambda x: x[2:-1] if x[0] == 'c' else x)
        train_set['RecipeIngredientQuantities'] = train_set['RecipeIngredientQuantities'].str.replace('\\','',)
        train_set['RecipeIngredientQuantities'] = train_set['RecipeIngredientQuantities'].str.replace('"','',)
        train_set['RecipeIngredientQuantities'] = train_set['RecipeIngredientQuantities'].str.split(',')

        omnivor_list = [x.lower() for x in omnivor_list]
        vegetarisch_list = [x.lower() for x in vegetarisch_list]

        def classify_recipe(row):
            words = [ingredient.strip().lower() for ingredient in row]
            
            if any(word in words for word in omnivor_list):
                return 0
            elif any(word in words for word in vegetarisch_list):
                return 1        
            else:
                return 2

        train_set["RecipeClass"] = train_set['RecipeIngredientParts'].apply(classify_recipe)

        
        # /--------------------------------


        # TODO Aditi


        # /--------------------------------


        # TODO Andres


        # /--------------------------------


        # TODO Stefan


        # /--------------------------------

        

        target = train_set["Like"]
        train_set = train_set.drop("Like",axis=1)

        train_set.to_csv("train_set.csv",",")

        target = target.to_numpy()
        train_set = train_set.to_numpy()

        print("data wrangled")


        return train_set, target



main()
