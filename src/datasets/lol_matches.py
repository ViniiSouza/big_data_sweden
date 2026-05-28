from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, DoubleType
from src.datasets.base import DatasetInfo, DatasetLoader


class LolMatchesLoader(DatasetLoader):
    """League of Legends Match Dataset 2025.

    Source: https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025

    Raw schema (94 columns): 
        game_id, game_start_utc, game_duration, game_mode, game_type, game_version,
        map_id, platform_id, queue_id, participant_id, puuid, summoner_name, summoner_id,
        summoner_level, champion_id, champion_name, team_id, win, ... (many more columns with in-game stats)

    Target: ``win`` (boolean) — binary classification of whether a team won the match.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="lol_matches",
            path="data/raw/lol_matches/league_data.csv",
            target_column="win",
            size_category="large",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        """Normalization decisions for this dataset:
        
        1. Column names are already snake_case in the CSV — nothing to rename.
        2. Several columns (36) are dropped as they are either IDs, redundant, or would leak the target.
           Leaving 58 columns that are candidate features or the target. 
        3. Explicit type casts so the downstream feature pipeline can rely on
           them (do not trust inferSchema across team members / configs).
        4. Drop rows with null target.
        5. Keep only the expected columns at the end (locks the output contract).
        """

        cols_to_drop = [
            # Ids
            "game_id", "game_start_utc", "map_id", "participant_id", "puuid", "summoner_id", "summoner_name",
            "summoner_level", "champion_id",
            # Metadata
            "platform_id", "game_mode", "game_type", "queue_id", "game_version",
            "individual_position", "lane", "role",
            # Time
            "game_duration",
            # Items & spell IDs
            "item0", "item1", "item2", "item3", "item4", "item5", "item6",

            "solo_tier", "solo_rank", "solo_lp", "solo_wins", "solo_losses",
            "flex_tier", "flex_rank", "flex_lp", "flex_wins", "flex_losses",

            "champion_mastery_lastPlayTime_utc"
        ]

        string_cols = [
            "champion_name", "team_position"
        ]

        long_cols = [
            "total_damage_dealt", "total_damage_dealt_to_champions",
            "physical_damage_dealt_to_champions",
            "magic_damage_dealt_to_champions",
            "true_damage_dealt_to_champions",
            "damage_dealt_to_objectives", "damage_dealt_to_turrets",
            "total_damage_taken", "physical_damage_taken",
            "magic_damage_taken", "true_damage_taken",
            "champion_mastery_points",
            "champion_mastery_pointsSinceLastLevel",
            "champion_mastery_pointsUntilNextLevel",
            "champion_mastery_lastPlayTime",
        ]

        integer_cols = [
            "team_id",

            "kills", "deaths", "assists",

            "baron_kills", "dragon_kills",

            "gold_earned", "gold_spent",

            "vision_score", "wards_placed", "wards_killed",
            "vision_wards_bought_in_game",

            "time_ccing_others",
            "champion_mastery_level",
            "champion_mastery_tokensEarned"

        ]

        double_cols = [
            "final_abilityHaste", "final_abilityPower", "final_armor",
            "final_armorPen", "final_armorPenPercent",
            "final_attackDamage", "final_attackSpeed",
            "final_bonusArmorPenPercent", "final_bonusMagicPenPercent",
            "final_ccReduction", "final_cooldownReduction",
            "final_health", "final_healthMax", "final_healthRegen",
            "final_lifesteal", "final_magicPen",
            "final_magicPenPercent", "final_magicResist",
            "final_movementSpeed", "final_omnivamp",
            "final_physicalVamp", "final_power",
            "final_powerMax", "final_powerRegen",
            "final_spellVamp"      
        ]

        target = self.info.target_column
        
        df = df.drop(*cols_to_drop)
        
        for col in integer_cols:
            df = df.withColumn(col, F.col(col).cast(IntegerType()))
        for col in long_cols:
            df = df.withColumn(col, F.col(col).cast(LongType()))
        for col in double_cols:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
        for col in string_cols + [target]:
            df = df.withColumn(col, F.trim(F.col(col).cast("string")))

        df = df.withColumn(
            target,
            F.when(F.col(target).isNull() | (F.col(target) == "NULL") | (F.col(target) == ""), "0")
            .otherwise(F.col(target))
        )

        df = df.filter(F.col(target).isin(["1", "0"]))
        

        return df.select(*string_cols, *long_cols, *integer_cols, *double_cols, target)
